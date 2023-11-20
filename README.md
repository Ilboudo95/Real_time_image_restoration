==========================================================================================================================================
MISE EN PLACE D’UNE SOLUTION DE RESTAURATION D’IMAGESEN TEMPS REEL POUR LA VIDEOSURVEILLANCE
==========================================================================================================================================

One of the important challenges in the field of computer vision, especially for automatic solutions such as detection, segmentation, recognition, monitoring, etc., is image quality. Image degradation, often caused by factors such as rain, fog, lighting,etc., lead to bad automatic decision making. 

Furthermore, various image restoration solutions exist, from restoration models for a single degradation to models for m ultipledegradations. However, these solutions are not suited for real-time processing.

In this study, our goal was to implement a real-time image restoration solution for video surveillance. To achieve this, using transfer learning with ResNet 50, we developed a model for the automatic identification of degradation types present in an image to determine the necessary treatments for image restoration. Our solution has the advantages of being flexible and scalable.

Furthermore, our solution is not recommended for the restoration of medical images and for others cases such as fire detection, smoke detection, and and domains where preserving fine details is crucial.


0. PROJECT STRUCTURE

    Restoration_project
        .requirements.txt
        .main.py
        .my_package.py

        ./modele
            .classif.pkl

            ./Deblurring
                .keras_metadata.pb  
                .saved_model.pb
                ./variables 
                .variables.data-00000-of-00001
                .variables.index

            ./Dehazing_indoor
                .keras_metadata.pb  
                .saved_model.pb
                ./variables 
                .variables.data-00000-of-00001
                .variables.index


            ./Dehazing_outdoor
                .keras_metadata.pb  
                .saved_model.pb
                ./variables 
                .variables.data-00000-of-00001
                .variables.index

            ./Denoising
                .keras_metadata.pb  
                .saved_model.pb
                ./variables 
                .variables.data-00000-of-00001
                .variables.index

            ./Deraining
                .keras_metadata.pb  
                .saved_model.pb
                ./variables 
                .variables.data-00000-of-00001
                .variables.index

            ./Enhancement
                .keras_metadata.pb  
                .saved_model.pb
                ./variables 
                .variables.data-00000-of-00001
                .variables.index

    -----------------------------directory Organigramm -----------------------------------------------------   

                            /Restoration_project
            ________________________|___________________________
            |                                                   |
            |                                                   |
            |                                                   |
    [main.py(server),module(my_package.py),                   /modele
    requirements.txt(for requirements) ]                 _______|________________________________                                
                                                        |                                       |
                                                    /weights                          [classif.pkl(classification modeles)]
                _________________________________________|_______________________________________________                           
                [/Deblurring; /Dehazing_indoor, /Dehazing_outdoor, /Denoising; /Deraining; /Enhancement ]

1- REQUIREMENTS

    All the requirements are specified in the requierements file named "requirements.txt"
    we are the following librairies:

    starlette==0.27.0
    fastapi==0.103.1
    tensorflow==2.10.1
    tensorflow-hub==0.13.0
    ISR==2.2.0
    Pillow==9.4.0
    fastai==2.7.13
    matplotlib==3.7.1
    numpy==1.25.0
    opencv-python== 4.8.0.76
    moviepy==1.0.3
    dill==0.3.6
    uvicorn==0.23.2

    run this line on cmd prompt:  pip install -r requirements.txt
2 - DEPLOYEMENT

    First, be in work directory ( "/Image_restoration") 
    run the code in the CMD prompt

    uvicorn main:app --host 0.0.0.0 --port 8000   (or use another free port)

3 - USE

    on the browser:

    localhost:8000 (or specified port)

    or for use from other laptop:
    ip_of_server_laptop:8000 (or specified port) ( be in the same network)

    in the home page:
        image: restore image, juste upload degraded image and click restore button

        video: restore video, juste upload degraded video and click restore button

4- Funtion DETAILS

    resize_image: for resizing images
    get_model: for loading models weights
    prediction_operation: for detecting degradations types 
    display_side_by_side_image : for displaying image side by side (original and restore)
    display_side_by_side: for display video side by side (original and restore)
