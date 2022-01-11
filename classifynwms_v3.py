# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__copyright__ = "Apache 2 license. Made by Multimedia University Cytomine Team, Cyberjaya, Malaysia, http://cytomine.mmu.edu.my/"
__version__ = "1.3.0"

# Date created: V3 - 21 December 2021

import sys
import numpy as np
import os
import argparse
import json
import logging
import cytomine
import shutil

from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob
from tifffile import imread
#from csbdeep.utils import Path, normalize
# from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, Project, ImageInstance, Property, JobData
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection
# from cytomine.models.property import Tag, TagCollection, PropertyCollection
# from cytomine.utilities.software import parse_domain_list, str2bool, setup_classify, stringify

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet201
pretrained_model = DenseNet201(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
pretrained_model.trainable = False

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2
import math
from pathlib import Path



def densemodel():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    ])
    tensor = tf.keras.Input((224, 224, 3))
    x = tf.cast(tensor, tf.float32)
    x = tf.keras.applications.densenet.preprocess_input(
        x, data_format=None)
    x = data_augmentation(x)
    x = pretrained_model(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(4)(x)
    x = tf.nn.softmax(x)
    model = tf.keras.Model(tensor, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def run(cyto_job, parameters):
    logging.info("----- test software v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project

    images = ImageInstanceCollection().fetch_with_filter("project", project.id)
    job.update(status=Job.RUNNING, progress=2, statusComment="Images gathered...")

    terms = TermCollection().fetch_with_filter("project", project.id)
    job.update(status=Job.RUNNING, progress=4, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()

    model_dir = "models"
    model_name = "weights.best_v10b_100ep_cc_LR_01val.h5"

    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = [int(id_img) for id_img in parameters.cytomine_id_images.split(',')]
        print(list_imgs)

    working_path = os.path.join("tmp", str(job.id))
    base_path=str(Path.home())
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, model_name)
#     model_path = os.path.join(base_path, model_dir, model_name)
    print("Model path:", model_path)
    
    print('Loading model.....')
    model = densemodel()
    model.load_weights(model_path)
    print('Model successfully loaded!')
    IMAGE_CLASSES = ['c0', 'c1', 'c2', 'c3']
    IMAGE_WIDTH, IMAGE_HEIGHT = (224, 224)

    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:
        
        id_project=project.id   
        output_path = os.path.join(working_path, "densenet201_results.csv")
        f= open(output_path,"w+")
        #Go over images
        job.update(status=Job.RUNNING, progress=10, statusComment="Running DenseNet201 classification on image...")
        #for id_image in conn.monitor(list_imgs, prefix="Running PN classification on image", period=0.1):
        for id_image in list_imgs:
            print('Current image:', id_image)
            roi_annotations = AnnotationCollection()
            roi_annotations.project = project.id
            roi_annotations.term = parameters.cytomine_id_cell_term
            roi_annotations.image = id_image #conn.parameters.cytomine_id_image
            roi_annotations.job = parameters.cytomine_id_annotation_job
            roi_annotations.user = parameters.cytomine_id_user_job
            roi_annotations.showWKT = True
            roi_annotations.fetch()
            print(roi_annotations)

            start_prediction_time=time.time()
            predictions = []
            img_all = []
            pred_all = []
            pred_c0 = 0
            pred_c1 = 0
            pred_c2 = 0
            pred_c3 = 0

            f.write("ID;Image;Project;Term;User;Area;Perimeter;WKT \n")

            #Go over ROI in this image
            #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
            # for roi in roi_annotations:
            for i, roi in enumerate(roi_annotations):
                #Get Cytomine ROI coordinates for remapping to whole-slide
                #Cytomine cartesian coordinate system, (0,0) is bottom left corner                
                print("----------------------------Cells------------------------------")
                roi_geometry = wkt.loads(roi.location)
                # print("ROI Geometry from Shapely: {}".format(roi_geometry))
                print("ROI Bounds")
                print(roi_geometry.bounds)
                minx=roi_geometry.bounds[0]
                miny=roi_geometry.bounds[3]
                #Dump ROI image into local PNG file
                # roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                print(roi_path)
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')
                job.update(status=Job.RUNNING, progress=20, statusComment=roi_png_filename)
                print("roi_png_filename: %s" %roi_png_filename)
                roi.dump(dest_pattern=roi_png_filename,alpha=True)

                im = cv2.imread(roi_png_filename)
                im_arr = np.array(im)
                im_arr = cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)
                im_arr = cv2.resize(im_arr, (224, 224))
                im_arr = np.expand_dims(im_arr, axis=0)

                predictions.append(model.predict(im_arr))
                pred_labels = np.argmax(predictions, axis=-1)
        
                # print("Prediction:", predictions)

                pred_labels = np.argmax(predictions, axis=-1)
                # print("PredLabels:", pred_labels)            
                img_all.append(roi_png_filename)
                # print(img_all)
                
                
                pred_all.append(pred_labels)
                # print(pred_all)

                # roi_class_path=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png')

                if pred_labels[i][0]==0:
                    # print("Class 0: Negative")
                    id_terms=parameters.cytomine_id_c0_term
                    pred_c0=pred_c0+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class0/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==1:
                    # print("Class 1: Weak")
                    id_terms=parameters.cytomine_id_c1_term
                    pred_c1=pred_c1+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class1/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==2:
                    # print("Class 2: Moderate")
                    id_terms=parameters.cytomine_id_c2_term
                    pred_c2=pred_c2+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class2/'+str(roi.id)+'.png'),alpha=True)
                elif pred_labels[i][0]==3:
                    # print("Class 3: Strong")
                    id_terms=parameters.cytomine_id_c3_term
                    pred_c3=pred_c3+1
                    # roi.dump(dest_pattern=os.path.join(roi_path+'Class3/'+str(roi.id)+'.png'),alpha=True)


                cytomine_annotations = AnnotationCollection()

                annotation=roi_geometry



                cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                       id_image=id_image,#conn.parameters.cytomine_id_image,
                                                       id_project=parameters.cytomine_id_project,
                                                       id_terms=[id_terms]))
                print(".",end = '',flush=True)

                #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                ca = cytomine_annotations.save()
                cytomine_annotations.project = project.id
                cytomine_annotations.job = job.id
                cytomine_annotations.user = user
                cytomine_annotations.showAlgo = True
                cytomine_annotations.showWKT = True
                cytomine_annotations.showMeta = True
                cytomine_annotations.showGIS = True
                cytomine_annotations.showTerm = True
                cytomine_annotations.annotation = True
                cytomine_annotations.fetch()
                print(cytomine_annotations)
                
                # print('cytomine_annotations ID: ',cytomine_annotations.id)

            # cytomine_annotations = AnnotationCollection()
            for annotation in cytomine_annotations:
                # print(annotation.id)
                # f.write("{};{};{};{};{};{};{};{}\n".format(annotation.id,annotation.image,annotation.project,annotation.term,annotation.user,annotation.area,annotation.perimeter,annotation.location))
                f.write("{};{};{};{};{};{};{};{}\n".format(annotation.id,annotation.image,annotation.project,annotation.term,annotation.user,annotation.area,annotation.perimeter,annotation.location))
        

            # print("prediction all:", pred_all)
            # print(pred_labels)

            # print("prediction c0:", pred_c0)
            # print("prediction c1:", pred_c1)
            # print("prediction c2:", pred_c2)
            # print("prediction c3:", pred_c3)
            pred_all=[pred_c0, pred_c1, pred_c2, pred_c3]
            pred_positive_all=[pred_c1, pred_c2, pred_c3]
            print("pred_all:", pred_all)
            im_pred = np.argmax(pred_all)
            print("image prediction:", im_pred)
            pred_total=pred_c0+pred_c1+pred_c2+pred_c3
            print("pred_total:",pred_total)
            pred_positive=pred_c1+pred_c2+pred_c3
            print("pred_positive:",pred_positive)
            print("pred_positive_all:",pred_positive_all)
            print("pred_positive_max:",np.argmax(pred_positive_all))
            pred_positive_100=pred_positive/pred_total*100
            print("pred_positive_100:",pred_positive_100)

            if pred_positive_100 == 0:
                proportion_score = 0
            elif pred_positive_100 < 1:
                proportion_score = 1
            elif pred_positive_100 >= 1 and pred_positive_100 <= 10:
                proportion_score = 2
            elif pred_positive_100 > 10 and pred_positive_100 <= 33:
                proportion_score = 3
            elif pred_positive_100 > 33 and pred_positive_100 <= 66:
                proportion_score = 4
            elif pred_positive_100 > 66:
                proportion_score = 5

            if pred_positive_100 == 0:
                intensity_score = 0
            elif im_pred == 0:
                intensity_score = np.argmax(pred_positive_all)
            elif im_pred == 1:
                intensity_score = 1
            elif im_pred == 2:
                intensity_score = 2
            elif im_pred == 3:
                intensity_score = 3

            allred_score = proportion_score + intensity_score
            print('Proportion Score: ',proportion_score)
            print('Intensity Score: ',intensity_score)            
            print('Allred Score: ',allred_score)
            
            
        end_time=time.time()
        print("Execution time: ",end_time-start_time)
        print("Prediction time: ",end_time-start_prediction_time)
        
        f.write(" \n")
        f.write("Class Prediction;Class 0 (Negative);Class 1 (Weak);Class 2 (Moderate);Class 3 (Strong);Total Prediction;Total Positive;Class Positive Max;Positive Percentage;Proportion Score;Intensity Score;Allred Score;Execution Time;Prediction Time \n")
        f.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(im_pred,pred_c0,pred_c1,pred_c2,pred_c3,pred_total,pred_positive,np.argmax(pred_positive_all),pred_positive_100,proportion_score,intensity_score,allred_score,end_time-start_time,end_time-start_prediction_time))
        
        f.close()
        job_data = JobData(job.id, "Generated File", "annotations.csv").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")

    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)


