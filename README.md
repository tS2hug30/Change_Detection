# Change Detection
Multi-tusk U-netを用いた変化点検出のソースコード

## Constitution
Change_Detection
      |
      -- Multi_task_u-net
              |
              -- GT/Original
              |
              -- Image_all ----- Annotation ---- t0
              |              |                |
              |              |                -- t1
              |              |
              |              --- Gray_ano ------ t0
              |              |                |
              |              |                -- t1
              |              |
              |              --- gray.py
              |              |
              |              --- Original ------ t0
              |              |                |
              |              |                -- t1
              |              |
              |              --- Test ---------- t0 --- Annotation
              |                               |      |
              |                               |      -- ORIGINAL
              |                               |
              |                               |- t1 --- Annotation
              |                                      |
              |                                      -- ORIGINAL
              |                 
              -- unet ----------- *.py
                             |
                             --- feat_visual --- 0 ~ 10
                             |                |
                             |                -- threshold
                             |
                             --- OUTPUT -------- image0,1
                             |                |
                             |                -- result0,1
                             |                |
                             |                -- true
                             |
                             --- TRAIN


## Usage
