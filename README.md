Due to the high density of buildings
and improper water drainage facilities, flash
floods are prevalent in cities. In such scenarios,
to extend timely aid, it is often required to
estimate the level of flood in different parts of
the affected zone. This paper presents a novel
method employing participatory sensing and
computer vision to estimate the flood level. The
method involves the participants capturing and
uploading images of the partially submerged
static structures such as buildings, lampposts
etc., using their smart phones or other
intelligent devices. The captured images are
geo-tagged and uploaded to a server. The
feature matching algorithm, SIFT finds the
corresponding matching feature points between
the captured and a reference image at the
server. The flood line is then estimated and
drawn against the reference image. To optimize
the network bandwidth usage, images with
average resolutions are used. This method
yields immediate and considerably accurate
results thus helping in isolating areas that have
been severely impacted for timely help. 





Requirments

absl-py==0.7.1 astor==0.8.0 attrs==19.1.0 backcall==0.1.0 bleach==3.1.0 certifi==2019.6.16 chardet==3.0.4 colorama==0.4.1 colorspacious==1.1.2 cycler==0.10.0 decorator==4.4.0 defusedxml==0.6.0 entrypoints==0.3 gast==0.2.2 google-pasta==0.1.7 grpcio==1.22.0 h5py==2.9.0 idna==2.8 imutils==0.5.2 ipykernel==5.1.1 ipython==7.6.1 ipython-genutils==0.2.0 ipywidgets==7.5.0 jedi==0.14.1 Jinja2==2.10.1 joblib==0.13.2 jsonschema==3.0.1 jupyter==1.0.0 jupyter-client==5.3.1 jupyter-console==6.0.0 jupyter-core==4.5.0 Keras==2.2.4 Keras-Applications==1.0.8 Keras-Preprocessing==1.1.0 kiwisolver==1.1.0 Markdown==3.1.1 MarkupSafe==1.1.1 matplotlib==3.1.1 mistune==0.8.4 nbconvert==5.5.0 nbformat==4.4.0 notebook==6.0.0 numpy==1.16.4 opencv-python==4.1.0.25 pandas==0.25.0 pandocfilters==1.4.2 parso==0.5.1 pickleshare==0.7.5 Pillow==6.1.0 plottools==0.2.0 prometheus-client==0.7.1 prompt-toolkit==2.0.9 protobuf==3.9.0 Pygments==2.4.2 PyJWT==1.7.1 pyparsing==2.4.1 pyrsistent==0.15.3 PySocks==1.7.0 python-dateutil==2.8.0 pytz==2019.1 pywinpty==0.5.5 PyYAML==5.1.1 pyzmq==18.0.2 qtconsole==4.5.1 requests==2.22.0 scikit-learn==0.21.2 scikit-plot==0.3.7 scipy==1.3.0 seaborn==0.9.0 Send2Trash==1.5.0 six==1.12.0 sklearn==0.0 tensorboard==1.14.0 tensorflow==1.15.0 tensorflow-estimator==1.14.0 termcolor==1.1.0 terminado==0.8.2 testpath==0.4.2 tornado==6.0.3 traitlets==4.3.2 twilio==6.29.1 urllib3==1.25.3 wcwidth==0.1.7 webencodings==0.5.1 Werkzeug==0.15.5 widgetsnbextension==3.5.0 wrapt==1.11.2
