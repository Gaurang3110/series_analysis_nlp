from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(df):
    class_weights = compute_class_weight("balanced",
                         classes=sorted(df['label'].unique().tolist()),
                         y = df['label'].tolist()
                         )
    return class_weights