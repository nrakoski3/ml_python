from HW1.autos.decision_tree import decision_tree_main


def main():
    decision_tree_main()


# run main() if file is called
if __name__ == "__main__":
    main()


# Soy Bean Trial Code
    # df, data, target = import_df.soybean_training()
    # print(df.head())
    # print(df.loc[df[0] == '09YT000052'])
    # print(df.loc[df['A'] == 'foo'])
    # print(df.loc[df['A'] == 'foo'])

    # Normalize 0,2,3,4
    # n = 0
    # temp_arr = []
    # num_arr = []
    # for ind, family in enumerate(data[:, 4]):
    #     test = True
    #     for idx, name in enumerate(temp_arr):
    #         if family == name:
    #             test = False
    #             num_arr.append(num_arr[idx])  # append copy's group number
    #             data[ind, 4] = num_arr[idx]
    #     if test:
    #         n += 1
    #         temp_arr.append(family)  # new family name, append to list
    #         num_arr.append(n)
    #         data[ind, 4] = n
    # print(num_arr.__len__())
    # # print(temp_arr)
    # print(data[:, 4].__len__())
    # print(data[:, 4])

