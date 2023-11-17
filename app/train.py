import argparse

from engine import TextClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop() 
    required = parser.add_argument_group('required arguments')
    required.add_argument('--csv', required=True)
    optional.add_argument('--columns', default='Main')
    optional.add_argument('--fract', default=1.0, type=float)
    optional.add_argument('--output-validation', const=True, nargs="?", default=False, type=bool)
    optional.add_argument('--output', default='/app/output')
    parser._action_groups.append(optional)

    args = parser.parse_args()

    print('Arguments given:')
    for arg, value in vars(args).items():
        print(f'  {arg}: {value}')
    print('')

    return args

def train(csv_file, columns, output, output_validation=False):
    classifier = TextClassifier()

    df = classifier.load_data(csv_file=csv_file, frac=args.fract)
    if not len(df):
        print(f'Failed to load data from "{csv_file}"')
        return

    print(f'{len(df)} rows loaded')

    texts, labels, train_texts, train_labels, test_texts, test_labels = classifier.make_data_sets(df, columns=columns)

    column_names = "_".join(columns).lower()
    df.to_csv(f"{output}/{column_names}_dl.csv", mode='w', columns=['Text','Label'], index=False)

    model = classifier.fit(train_texts, train_labels)

    classifier.export_model(f"{output}/{column_names}_model.pkl")

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
    classifier.pickle(slugs, f"{output}/{column_names}_slugs.pkl")

    classifier.validate_model(
        test_texts=test_texts,
        test_labels=test_labels,
        dst_file=f"{output}/{column_names}-matrix.pdf",
        dst_csv=f"{output}/{column_names}-matrix.csv",
        dst_validation = f"{output}/{column_names}_validation.csv" if output_validation else None
    )


if __name__ == '__main__':
    args = parse_args()
    train(args.csv, args.columns.split(','), args.output, args.output_validation)
