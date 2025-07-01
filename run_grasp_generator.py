from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='/home/leo/code/二区/vmgnet/logs/epoch_01_iou_0.89',
        visualize=True
    )
    generator.load_model()
    generator.run()
