import os
import dicom2nifti
from io import BytesIO
import pydicom
import tempfile
import pandas as pd
import numpy as np
import yaml
from importlib import import_module

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from step1_main import savenpy
from data_loader import DataBowl3Detector, collate
from utils import (
    getFreeId,
    setgpu,
    Logger,
    split4,
    combine4,
    split8,
    combine8,
    split16,
    combine16,
    split32,
    combine32,
    split64,
    combine64,
)
from split_combine import SplitComb
from test_detect import test_detect
from data_classifier import DataBowl3Classifier
from model import seed_everything, MultipathModel, MultipathModelBL
import pdb


class MDAIModel:
    def __init__(self):
        self.input_dir = tempfile.mkdtemp()
        self.prep_dir = tempfile.mkdtemp()
        self.bbox_dir = tempfile.mkdtemp()
        self.feat_dir = tempfile.mkdtemp()

        self.nod_net_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "2_nodule_detection",
            "detector.ckpt",
        )
        self.case_net_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "3_feature_extraction",
            "classifier_state_dictpy3.ckpt",
        )
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "4_co_learning", "pretrain.pth",
        )

        self.df = pd.read_csv(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_dcm.csv")
        )

        with open(os.path.join(os.path.dirname(__file__), "params_config.yaml")) as f:
            self.cfig = yaml.load(f)

    def predict(self, data):
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs = []
        dicom_files = []
        for file in input_files:
            if file["content_type"] != "application/dicom":
                continue

            dicom_files.append(pydicom.dcmread(BytesIO(file["content"])))

        self.df = self.df[self.df["exam_id"] == dicom_files[0].SeriesInstanceUID]
        sess_splits = self.df["exam_id"].tolist()
        sess_id = sess_splits[0]

        # 1. Convert to nifti
        nifti_file = dicom2nifti.convert_dicom.dicom_array_to_nifti(
            dicom_files,
            output_file=os.path.join(self.input_dir, sess_id + ".nii.gz"),
            reorient_nifti=True,
        )
        print("1. Converted to nifti")

        # 2. Preprocess image
        savenpy(
            name=sess_id,
            prep_folder=self.prep_dir,
            data_path=os.path.join(self.input_dir, sess_id + ".nii.gz"),
        )
        np.save(self.prep_dir + "/" + sess_id + "_label.npy", np.zeros((1, 4)))
        os.remove(
            os.path.join(self.input_dir, sess_id + ".nii.gz")
        )  # remove nifti files
        print("2. Preprocessed images!")

        # 3. Nodule detection
        config = self.cfig["detect"]
        config["datadir"] = self.prep_dir
        config["testsplit"] = sess_splits
        nodmodel = import_module("net_detector")
        config1, nod_net, _, get_pbb = nodmodel.get_model()
        config1["datadir"] = config["datadir"]
        config1["gpu"] = config["gpu"]
        nod_net.load_state_dict(torch.load(self.nod_net_path)["state_dict"])
        nod_net = nod_net

        split_comber = SplitComb(
            config1["sidelen"],
            config1["max_stride"],
            config1["stride"],
            config1["margin"],
            pad_value=config1["pad_value"],
        )
        dataset = DataBowl3Detector(
            config["testsplit"], config1, phase="test", split_comber=split_comber
        )
        del split_comber
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate,
        )

        del dataset

        test_detect(test_loader, nod_net, get_pbb, self.bbox_dir, config1)
        del test_loader
        del nod_net
        print("3. Nodule detection completed!")

        # 4. Feature extraction
        config2 = self.cfig["cls"]
        casemodel = import_module("net_classifier")
        casenet = casemodel.CaseNet(topk=5)
        state_dict = torch.load(self.case_net_path)

        casenet.load_state_dict(state_dict)
        casenet = casenet  # .cuda()

        config2["bboxpath"] = self.bbox_dir
        config2["datadir"] = self.prep_dir
        config2["feat128_root"] = self.feat_dir

        testsplit = sess_splits

        def test_casenet(model, testset):
            data_loader = DataLoader(
                testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False
            )

            model.eval()
            predlist = []
            device = torch.device(
                "cuda"
                if torch.cuda.is_available() and self.cfig["cls"]["gpu"]
                else "cpu"
            )

            for i, (x, coord, subj_name) in enumerate(data_loader):
                print(i, subj_name[0])
                coord = Variable(coord).to(device)  # .cuda()
                x = Variable(x).to(device)  # .cuda()
                model = model.to(device)
                _, casePred, feat128, _ = model(x, coord)
                predlist.append(casePred.data.cpu().numpy())
                # print (out.data.cpu().numpy().shape, out[0].data.cpu().numpy().shape)
                fname128 = config2["feat128_root"] + "/" + subj_name[0] + ".npy"
                np.save(fname128, feat128.cpu().data.numpy())

            predlist = np.concatenate(predlist)
            return predlist

        dataset = DataBowl3Classifier(testsplit, config2, phase="test")
        predlist = test_casenet(casenet, dataset).T
        print("4. Feature extraction completed!")

        # 5. Classification
        need_factor = [
            "with_image",
            "with_marker",
            "age",
            "education",
            "bmi",
            "phist",
            "fhist",
            "smo_status",
            "quit_time",
            "pkyr",
            "plco",
            "kaggle_cancer",
        ]
        sess_mark_dict = {}
        testsplit = sess_splits

        for i, item in self.df.iterrows():
            test_biomarker = np.zeros(12).astype("float32")
            for j in range(len(need_factor)):
                test_biomarker[j] = item[need_factor[j]]
            sess_mark_dict[item["exam_id"]] = test_biomarker
        data_path = self.feat_dir

        model = MultipathModelBL(1)
        model.load_state_dict(torch.load(self.model_path))

        patient_list = []
        exam_list = []
        pred_list = []
        for i, item in self.df.iterrows():
            pid = item["patient_id"]
            sess_id = item["exam_id"]
            patient_list.append(pid)
            exam_list.append(sess_id)
            test_biomarker = sess_mark_dict[sess_id]

            test_imgfeat = np.load(data_path + "/" + sess_id + ".npy")
            test_biomarker = torch.from_numpy(test_biomarker).unsqueeze(0)
            test_imgfeat = torch.from_numpy(test_imgfeat).unsqueeze(0)
            _, _, _, _, bothPred = model(
                test_imgfeat, test_biomarker, test_imgfeat, test_biomarker
            )
            pred_list += list(bothPred.data.cpu().numpy())

        pred_list = ["{:.4f}".format(prob) for prob in pred_list]
        outputs = [
            {
                "type": "ANNOTATION",
                "study_uid": str(dicom_files[0].StudyInstanceUID),
                "series_uid": str(dicom_files[0].SeriesInstanceUID),
                "instance_uid": str(dicom_files[0].SOPInstanceUID),
                "class_index": 0 if float(pred_list[0]) >= 0.5 else 1,
                "probability": float(pred_list[0]),
            }
        ]
        print("5. Classification completed!")
        return outputs
