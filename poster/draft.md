1) welcome to my presentation of + title
2) present the problem (student copies or pays attention but cant do both at the same time)
3) our project tries to solve this problem transcribing in real time what the professor writes to free the student from this burden
4) our solution is built on a 10 stages real time pipeline
5) VIDEO FEED: (represents maybe is not best verb) a video stream that come from the camera that points to the whiteboard.
6) BOARD SEGMENTATION: is responsible to identify region that represents the board.
7) PERSON SEGMENTATION: does the same but for the person
8) PERSPECTIVE CORRECTION: since the camera is not always ideally positioned in the center of the scene -> next stage is to correct the perspective + crop out of scene to maximize accuracy 
9) SURFACE RECONSTRUCTION has the job to create a new virtual memory of the board without the professor using output of the person segmentation. At every input frame we update the memory of the board using the regions where the person is not present
10) TEXT LINE DETECTION: once we have a clean unoccluded and rectified perspective of the board, we can begin to analyze the text, we start detecting regions containing text
11) BLOCK GROUPING: is just grouping text regions together into paragraphs
12) ENTITY REGISTRY: once we have blocks, each one is inserted in a registry to monitor the evolution of the content over (the) time. IMPORTANT: THIS STAGE IS THE CORE WHICH DEFINES WHAT WILL BE TRANSCRIPTED
13) LEDGER SYNTHESIS: the evolution of the blocks in the registry have the following states, upgrade of states, to elect a block text as valid for transcription
14) the final stage archives all the blocks accumulated over the entire video stream to a file called lecture_history.md + current blocks are shown in live.md


poster improvements:
- add self arrow to Board Segmentation
- add numbers to stages
- add arrow from person segmentation to board reconstructor
- add arrows INFERRING -> STAGE OCR -> ACTIVE
- make block states a zoom out of the registry
- colors in the pipeline are wrong
- legdger -> archiver
- entity registry -> more like a tracker, or monitor
- would have been nice to show board with corners and person with mask
