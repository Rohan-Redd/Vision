# VISION  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <img src="https://github.com/Rohan-Redd/Vision/blob/main/Static/img/fav.png" width="100" height="70"> 
Object Detection and Face Recognition Website

According to the World Health Organization, roughly 2.2 billion people are visually impaired. Vision loss can cause problems in everyday life, but with the help of 'VISION,' we could make the lives easier.

'VISION' assists visually challenged people by acting as their virtual vision. Keyboard keys can be used to browse the website. The items in front of the individual can be detected using a camera. The faces of others in front of user can also be recognised, and an audio output is sent to the user.

## Installation

### Requirements
* Python 3.10
* Preferably Windows Operating system
* Camera

### Libraries Required
```
numpy
opencv-python
cmake
dlib
face_recognition
flask
```


If dlib installation does not work, install the python wheel,
```
https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl 
```

## How to Run?

* After the requirements are matched, run [app.py](https://github.com/Rohan-Redd/Vision/blob/main/app.py)
* Website can be accessed on local system at http://127.0.0.1:5000
* *Optional - images in the [images](https://github.com/Rohan-Redd/Vision/tree/main/Images) folder are the known faces whom the face recognition can recognize. So they can be changed as pleased

## Website Links
* Home Page - http://127.0.0.1:5000
* Object Detection Page - http://127.0.0.1:5000/obj_detect
* Face Recognition Page - http://127.0.0.1:5000/face_recog
* About Page - http://127.0.0.1:5000/about

## Keyboard Controls
* Home Page 
```
   * Press key 'F' to go to Object Detection Page
   * Press key 'J' to go to Face Recognition Page
```

* Object Detection Page
```
   * Press key 'F' to go to Home Page
   * Press key 'J' to go to Face Recognition Page
```

* Face Recognition Page
```
   * Press key 'F' to go to Home Page
   * Press key 'J' to go to Object Detection Page
```

* About Page 
```
   * Press key 'F' to go to Object Detection Page
   * Press key 'J' to go to Face Recognition Page
```
