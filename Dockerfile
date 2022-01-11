# * Copyright (c) 2009-2020. Authors: see NOTICE file.
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

FROM cytomine/software-python3-base

#INSTALL
# RUN apt-get update
RUN pip install tensorflow
RUN pip install keras
RUN pip install matplotlib
RUN pip install numpy
RUN pip install shapely
RUN pip install tifffile
RUN pip install h5py
# RUN pip install pathlib

# RUN mkdir -p /models 
# ADD weights.best_v10b_100ep_cc_LR_01val.h5 /models/weights.best_v10b_100ep_cc_LR_01val.h5
# RUN chmod 444 /models/weights.best_v10b_100ep_cc_LR_01val.h5


RUN mkdir -p /app
ADD weights.best_v10b_100ep_cc_LR_01val.h5 /app/weights.best_v10b_100ep_cc_LR_01val.h5
ADD descriptor.json /app/descriptor.json
ADD classifynwms_v3.py /app/classifynwms_v3.py

ENTRYPOINT ["python3", "/app/classifynwms_v3.py"]
