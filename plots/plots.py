import matplotlib.pyplot as plt
import pandas as pd

def plot_mortality(cancer_type):
    PATH_mortality = '../datasets/clean_datasets/mortality_clean'
    mor = pd.read_csv(PATH_mortality + ".csv")
    mor = mor[mor.type == cancer_type]
    dico_mor = {}

    for i in range(len(mor)):
        country = mor.iloc[i]['area']
        year = int(mor.iloc[i]['year'])
        true_mor = mor.iloc[i]['sum']
        if country not in dico_mor.keys():
            dico_mor[country] = [(year, true_mor)]
        else:
            dico_mor[country] += [(year, true_mor)]

    max_mors = {}
    for k, v in dico_mor.items():
        max_mors[k] = max([mor for _, mor in v])

    mor_countries = list(dico_mor.keys())
    mor_countries.sort()
    print(mor_countries)

    PATH_wb = '../datasets/clean_datasets/Worldbank_Replaced_Countries'
    wb = pd.read_csv(PATH_wb + ".csv")
    dico_wb = {}
    for i in range(len(wb)):
        country = wb.iloc[i]['area']
        year = int(wb.iloc[i]['year'])
        true_mor = 0
        if country not in dico_wb.keys():
            dico_wb[country] = [(year, true_mor)]
        else:
            dico_wb[country] += [(year, true_mor)]

    wb_countries = list(dico_wb.keys())
    wb_countries.sort()
    print(wb_countries)

    PATH_fao = '../datasets/clean_datasets/FAOSTAT_Replaced_Countries'
    fao = pd.read_csv(PATH_fao + ".csv")
    dico_fao = {}
    for i in range(len(fao)):
        country = fao.iloc[i]['area']
        year = int(fao.iloc[i]['year'])
        if country in max_mors.keys():
            true_mor = max_mors[country]
        else:
            true_mor = 10
        if country not in dico_fao.keys():
            dico_fao[country] = [(year, true_mor)]
        else:
            dico_fao[country] += [(year, true_mor)]

    fao_countries = list(dico_fao.keys())
    fao_countries.sort()
    print(fao_countries)

    for k, v in dico_mor.items():
        if len(v) > 1 :
            v.sort(key = lambda x : x[0])
        years = [year for year, _ in v]
        mors = [mor for _, mor in v]
        plt.scatter(years, mors, c='b')
        if k in dico_wb.keys():
            v_wb = dico_wb[k]
            if len(v_wb) > 1:
                v.sort(key=lambda x: x[0])
            years_wb = [year+5 for year, _ in v_wb]
            mors_wb = [mor for _, mor in v_wb]
            plt.scatter(years_wb, mors_wb, c='r')
        if k in dico_fao.keys():
            try:
                v_fao = dico_fao[k]
                if len(v_fao) > 1:
                    v_fao.sort(key=lambda x: x[0])
                years_fao = [year+5 for year, _ in v_fao]
                mors_fao = [mor for _, mor in v_fao]
                print(years_fao)
                print(mors_fao)
                plt.scatter(years_fao, mors_fao, c='g')
            except ValueError:
                pass

    # if k in dicolag.keys():
    #     v_lag = dicolag[k]
    #     years_lag = [year for year, _ in v_lag]
    #     mors_lag = [mor for _, mor in v_lag]
    #     plt.scatter(years_lag, mors_lag, c='r')
    # if len(years)>3:
    #     x_new = np.linspace(years[0], years[len(years)-1], 300)
    #     # print(years)
    #     # print(mors)
    #     # print(x_new)
    #     mors_smooth = spline(years, mors, x_new)
    #     plt.plot(x_new, mors_smooth)
        plt.savefig('../plots/evolution_per_country_with_other_data/' + k + '.png')
        plt.close()

plot_mortality('C16')