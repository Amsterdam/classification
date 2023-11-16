import argparse
from engine import TextClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop() 
    required = parser.add_argument_group('required arguments')
    required.add_argument('--csv', required=True)
    optional.add_argument('--columns', default='')
    optional.add_argument('--fract', default=1.0, type=float)
    optional.add_argument('--output-validation', const=True, nargs="?", default=False, type=bool)
    parser._action_groups.append(optional)
    return parser.parse_args()

def train(df, columns, output_validation=False):
    texts, labels, train_texts, train_labels, test_texts, test_labels = classifier.make_data_sets(df, columns=columns)
    colnames = "_".join(columns).lower()
    df.to_csv(f"/output/{colnames}_dl.csv", mode='w', columns=['Text','Label'], index=False)
    print(f"Training... for columns: {colnames}")
    model = classifier.fit(train_texts, train_labels)
    print("Serializing model to disk...")
    classifier.export_model(f"/output/{colnames}_model.pkl")

    if len(columns) > 1:
        categories = [
            x.split('|')
            for x in model.classes_
        ]

        slugs = [
            f"/categories/{category[0]}/sub_categories/{category[1]}"
            for category in categories
        ]
    else:
        slugs = [
            f"/categories/{category}"
            for category in model.classes_
        ]
    print(slugs)
    classifier.pickle(slugs, f"/output/{colnames}_slugs.pkl")
    
    print("Validating model")
    test_predict, precision, recall, accuracy = classifier.validate_model(
        test_texts,
        test_labels,
        f"/output/{colnames}-matrix.pdf",
        f"/output/{colnames}-matrix.csv",
        dst_validation = f"/output/{colnames}_validation.csv" if output_validation else None)
    print('Precision', precision)
    print('Recall', recall)
    print('Accuracy', accuracy)


if __name__ == '__main__':
    args = parse_args()
    print("Using args: {}".format(args))

    classifier = TextClassifier()
    df = classifier.load_data(csv_file=args.csv, frac=args.fract)
    if len(df) == 0:
        print("Failed to load {}".format(args.csv))
        exit(-1)
    else:
        print("{} rows loaded".format(len(df)))
    texts, labels, train_texts, train_labels, test_texts, test_labels = classifier.make_data_sets(df)
    columns = args.columns or 'Main'
    print("Training using category column(s): {}".format(columns))
    # train sub cat
    train(df, columns.split(','), args.output_validation)
