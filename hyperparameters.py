def get_hyperparameters(num_attributes=40):
    hyperparameters={
        'targets': ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows','Attractive',
                    'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                    'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                    'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                    'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                    'Wearing_Necktie', 'Young'],

        'height': 218, 
        'width': 178,
        'channels': 3, 
        'batch_size': 16, 
        'epochs': 1, 
        'num_tasks': 2, 
        'initializer': 'he_uniform', 
        'reg_lambda': 1e-3, 
        'output': [2]*num_attributes, 
        'lr': 1e-4, 
        'dropout_prob': 0.3,
    }
    
    return hyperparameters
