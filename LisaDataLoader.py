from sklearn.model_selection import train_test_split

from data_loader import DataLoader


class LisaDataLoader(DataLoader):
    def split_data(self, test_size=0.2, random_state=42) -> tuple:
        splitting_dict = {}
        df_types = ['filled_c', 'dropped_v']
        split_tts = 'split_tts'
        split_time = 'split_timebased'
        split_types = [split_tts, split_time]
        for df_selected, df_ in zip(df_types, self.train_data):
            for split_type in split_types:
                df = df_.dropna()
                if split_type == split_tts:
                    X_train, X_test, y_train, y_test = train_test_split(
                        df.drop(['is_click', 'day'], axis=1),
                        df['is_click'],
                        test_size=0.3,
                        random_state=100
                    )
                elif split_type == split_time:
                    num_train_days = round(df['day'].nunique() * 0.7)
                    train_days = df['day'].unique()[:num_train_days]
                    test_days = df['day'].unique()[num_train_days:]

                    df_train = df[df['day'].isin(train_days)]
                    df_test = df[df['day'].isin(test_days)]

                    X_train, y_train = df_train.drop(['is_click', 'day'], axis=1), df_train['is_click']
                    X_test, y_test = df_test.drop(['is_click', 'day'], axis=1), df_test['is_click']

                if split_type not in splitting_dict:
                    splitting_dict[split_type] = {}
                if df_selected not in splitting_dict[split_type]:
                    splitting_dict[split_type][df_selected] = {}

                for df_type, X, y in zip(['train', 'test'], [X_train, X_test], [y_train, y_test]):
                    X.rename(columns={'var1': 'action_count', 'webpage_id': 'webpage_id_count'}, inplace=True)

                    splitting_dict[split_type][df_selected][df_type] = {
                        'X': X,
                        'y': y
                    }
        return X_train, X_test, y_train, y_test