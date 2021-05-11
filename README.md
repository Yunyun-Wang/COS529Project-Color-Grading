## Eval single image

``python eval.py --load_from no_ref_32points/newest.ckpt --lut img_with_lut_applied.jpg --input your_img.jpg --output_path output_img.jpg
``

``python eval.py --load_from no_ref_32points_clearlut/newest.ckpt --clear_lut True --input your_img.jpg --output_path output_img.jpg
``