
Change_Detection --- Multi_task_u-net --- GT(MD) -------- Original(MD) /
                                       |
                                       -- Image_all ----- Annotation(MD) ---- t0(MD)
                                       |              |                    |
                                       |              |                    -- t1(MD)
                                       |              |
                                       |              --- Gray_ano(MD) ------ t0(MD)
                                       |              |                    |
                                       |              |                    -- t1(MD)
                                       |              |
                                       |              --- gray.py
                                       |              |
                                       |              --- Original(MD) ------ t0(MD)
                                       |              |                    |
                                       |              |                    -- t1(MD)
                                       |              |
                                       |              --- Test(MD) ---------- t0(MD) --- Annotation(MD)
                                       |                                   |          |
                                       |                                   |          -- ORIGINAL(MD)
                                       |                                   |
                                       |                                   |- t1(MD) --- Annotation(MD)
                                       |                                              |
                                       |                                              -- ORIGINAL (MD)
                                       |                 
                                       -- unet ----------- *.py
                                                      |
                                                      --- feat_visual(MD) --- 0 ~ 10(MD)
                                                      |                    |
                                                      |                    -- threshold(MD)
                                                      |
                                                      --- OUTPUT(MD) -------- image0,1(MD)
                                                      |                    |
                                                      |                    -- result0,1(MD)
                                                      |                    |
                                                      |                    -- true(MD)
                                                      |
                                                      --- TRAIN(MD)
