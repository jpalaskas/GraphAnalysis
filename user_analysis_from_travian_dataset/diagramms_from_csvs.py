import pandas as pd
import os
import glob


def read_csv_files():
    # cwd = os.fsencode(os.getcwd())
    # dirP = 'C:/Users/leonv/Desktop/Pttuxiakh'
    # Reading csv files repetively and insert to dataframes
    file_names = glob.glob("../data_users_moves/*.csv")
    attacks = pd.DataFrame(columns=['Timestamp', 'active', 'passive'])
    trades = pd.DataFrame(columns=['Timestamp', 'active', 'passive'])
    messages = pd.DataFrame(columns=['Timestamp', 'active', 'passive'])
    all_data = pd.DataFrame(columns=['Timestamp', 'active', 'passive'])

    for file in file_names:

        print(type(file))

        print('Currently using file - ', file)
        data_file1 = pd.read_csv(file, header=None)
        data_file1.columns = ['Timestamp', 'active', 'passive']
        # data_file2 = pd.read_csv(file, header=None)
        # data_file3 = pd.read_csv(file, header=None)
        if 'attack' in file:
            attacks = pd.concat([attacks, data_file1])
        elif 'trades' in file:
            trades = pd.concat([trades, data_file1])
        else:
            messages = pd.concat([messages, data_file1])
    all_data = pd.concat([data_file1, attacks, trades, messages])
    print(attacks, trades, messages)


def interactions_per_user(attacks, trades, messages):
    active_attacks_counts = attacks[['active', 'passive']].groupby(['active']).agg(['count'])
    active_trades_counts = trades[['active', 'passive']].groupby(['active']).agg(['count'])
    active_messages_counts = messages[['active', 'passive']].groupby(['active']).agg(['count'])
    passive__attacks_counts = attacks[['active', 'passive']].groupby(['passive']).agg(['count'])
    passive_trades_counts = trades[['active', 'passive']].groupby(['passive']).agg(['count'])
    passive__messages_counts = messages[['active', 'passive']].groupby(['passive']).agg(['count'])
    return active_attacks_counts, active_trades_counts, active_messages_counts, passive__attacks_counts, \
           passive_trades_counts, passive__messages_counts


def find_best_active_user(active_attacks_counts, active_trades_counts, active_messages_counts):
    popular_in_attacks = active_attacks_counts.loc[active_attacks_counts['active']['count'].idxmax()]
    popular_in_trades = active_trades_counts.loc[active_trades_counts['passive']['count'].idxmax()]
    popular_in_messages = active_messages_counts.loc[active_messages_counts['passive']['count'].idxmax()]
    return popular_in_attacks, popular_in_trades, popular_in_messages


def find_best_passive_user(passive__attacks_counts, passive_trades_counts, passive__messages_counts):
    popular_in_attacks_passive = passive__attacks_counts.loc[passive__attacks_counts['active']['count'].idxmin()]
    popular_in_trades_passive = passive_trades_counts.loc[passive_trades_counts['active']['count'].idxmin()]
    popular_in_messages_passive = passive__messages_counts.loc[passive__messages_counts['active']['count'].idxmin()]
    return popular_in_attacks_passive, popular_in_trades_passive, popular_in_messages_passive


def main():
    print("Reading---csv---files")
    read_csv_files()
    # print(attacks)
    # print(trades)
    # print(messages)

    # interactions_per_user(attacks, trades, messages)
    # active_attacks_counts, active_trades_counts, active_trades_counts, passive__attacks_counts, passive_trades_counts, passive__messages_counts = interactions_per_user(
    #     attacks, trades, messages)
    # best_active_user = find_best_active_user(active_attacks_counts, active_trades_counts, active_trades_counts,
    #                                          passive__attacks_counts, passive_trades_counts, passive__messages_counts)
    # best_passive_user = find_best_passive_user(active_attacks_counts, active_trades_counts, active_trades_counts,
    #                                            passive__attacks_counts, passive_trades_counts, passive__messages_counts)
