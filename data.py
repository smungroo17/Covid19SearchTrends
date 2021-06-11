import copy
import pandas as pd


class Data:
    def __init__(self):
        self.d = None
        self.import_data()
        self.filter_out_zeros()

    def import_data(self):
        hosp = pd.read_csv("hospitalization.csv")
        hosp = hosp[["open_covid_region_code", "date", "hospitalized_new"]]

        hosp["date"] = pd.to_datetime(
            hosp["date"], format="%Y-%m-%d"
        ) - pd.to_timedelta(7, unit="d")
        hosp = (
            hosp.groupby(
                [
                    "open_covid_region_code",
                    pd.Grouper(key="date", freq="W-MON", closed="left"),
                ]
            )
            .sum()
            .reset_index()
        )

        search = pd.read_csv("2020_US_weekly_symptoms_dataset.csv")
        search["date"] = pd.to_datetime(search["date"], format="%Y-%m-%d")

        self.d = pd.merge(
            search, hosp, how="left", on=["open_covid_region_code", "date"]
        )

    def fill_na(self, val):
        self.d = self.d.fillna(val)

    def filter_low_data_regions(self, threshold=0.1):
        """
        Keep all open_covid_region_code regions that have > 0.1 * #_elements that are not
        NaN values.
        :param threshold: cutoff point. Any region with less than 0.1 of all cells filled
                with data are filtered out
        """
        data_per_region = (
            self.d.groupby(["open_covid_region_code"]).count().iloc[:, 6:-1].sum(axis=1)
        )
        size_per_region = (
            self.d.groupby(["open_covid_region_code"]).size()[0] * self.d.shape[1]
        )
        good_regions = list(
            data_per_region[data_per_region > threshold * size_per_region].index
        )
        self.d = self.d[
            self.d["open_covid_region_code"].isin(good_regions)
        ].reset_index(drop=True)

    def filter_no_covid_cases(self):
        self.d = self.d.groupby(["open_covid_region_code"]).filter(
            lambda g: g["hospitalized_new"].sum() > 0
        )

    def filter_out_zeros(self):
        self.d = self.d.dropna(axis=1, how="all")

    def keep_x_symptoms(self, x):
        # sum all columns(symptoms)
        trim_data = copy.deepcopy(self.d)
        cropped = trim_data.loc[:, "symptom:Adrenal crisis":"symptom:Yawn"]
        s = cropped.sum(axis=0)

        # get 15 symptoms with highest sum value(sum of popularity)
        arr = [0] * x
        for j in range(x):
            # most common symptoms with sum of columns:
            arr[j] = s.idxmax()
            s = s.drop(s.idxmax(), axis=0)

        # keep only 15 symptoms with highest popularity and convert to csv
        self.d = trim_data.drop(
            columns=[
                col
                for col in trim_data.loc[:, "symptom:Adrenal crisis":"symptom:Yawn"]
                if col not in arr
            ]
        )

    def merge_regions(self):
        self.d = self.d.groupby(["date"]).sum().reset_index()

    def normalize_regions(self):
        median = self.d.groupby(["open_covid_region_code"]).mean().mean(axis=1)
        self.d = self.d.apply(
            lambda row: row.iloc[:6]  # preserve first 6 columns
            .append(row.iloc[6:-1] / median[row["open_covid_region_code"]])  # normalize features per region
            .append(row.iloc[-1:]),  # add labels
            axis=1,
        )
