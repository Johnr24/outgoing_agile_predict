import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import os

from django.core.management.base import BaseCommand, CommandError
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData, Nordpool, UpdateErrors

from config.utils import *
from config.settings import GLOBAL_SETTINGS

DAYS_TO_INCLUDE = 30
MODEL_ITERS = 50
MIN_HIST = 7
MAX_HIST = 80



class Command(BaseCommand):
    def add_arguments(self, parser):
        # Positional arguments
        # parser.add_argument("poll_ids", nargs="+", type=int)

        # Named (optional) arguments
        parser.add_argument(
            "--no_forecast",
            action="store_true",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
        )

        parser.add_argument(
            "--no_dow",
            action="store_true",
        )

        parser.add_argument(
            "--no_hist",
            action="store_true",
        )

        parser.add_argument(
            "--drop_last",
        )

        parser.add_argument(
            "--min_hist",
        )

        parser.add_argument(
            "--max_hist",
        )

        parser.add_argument(
            "--iters",
        )

        parser.add_argument(
            "--ignore_forecast",
            action="append",
        )

        parser.add_argument(
            "--reference_date",
            help="Run the update as if from this date (format: YYYY-MM-DD). If not provided, uses current date.",
        )

    def handle(self, *args, **options):
        debug = options.get("debug", False)
        reference_date = options.get("reference_date")
        
        if reference_date:
            try:
                reference_time = pd.Timestamp(reference_date, tz="GB")
                # Monkey patch Timestamp.now() for the duration of this run
                original_now = pd.Timestamp.now
                pd.Timestamp.now = lambda tz=None: reference_time
            except ValueError as e:
                raise CommandError(f"Invalid reference_date format. Please use YYYY-MM-DD. Error: {e}")

        no_forecast = options.get("no_forecast", False)
        no_hist = options.get("no_hist", False)

        drop_cols = ["total_wind"]
        if options.get("no_dow", False):
            drop_cols += ["day_of_week"]

        drop_last = int(options.get("drop_last", 0) or 0)

        min_hist = int(options.get("min_hist", MIN_HIST) or MIN_HIST)
        max_hist = int(options.get("max_hist", MAX_HIST) or MAX_HIST)

        iters = int(options.get("iters", MODEL_ITERS) or MODEL_ITERS)
        if options.get("ignore_forecast", []) is None:
            ignore_forecast = []
        else:
            ignore_forecast = [int(x) for x in options.get("ignore_forecast", [])]

        UpdateErrors.objects.all().delete()

        # Clean any invalid forecasts
        for f in Forecasts.objects.all():
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            if debug:
                print(f.id, f.name, q.count(), a.count())
            if q.count() < 600 or a.count() < 8000:
                f.delete()

        hist = get_history_from_model()

        print("Getting history from model")
        if debug:
            print("Getting history from model")
            print("Database History:")
            print(hist)
        start = pd.Timestamp("2023-07-01", tz="GB")
        if len(hist) > 48:
            hist = hist.iloc[:-48]
            start = hist.index[-1] + pd.Timedelta("30min")
            # for h in History.objects.filter(date_time__gte=start):
            #     h.delete()
        else:
            hist = pd.DataFrame()

        if debug:
            print(f"New data from {start.strftime(TIME_FORMAT)}:")

        new_hist, missing_hist = get_latest_history(start=start)

        # if len(missing_hist) > 0:
        #     print(">>> ERROR: Unable to update history due to missing columns:", end="")
        #     for c in missing_hist:
        #         print(c, end="")
        #         obj=UpdateErrors(
        #             date_time=pd.Timestamp.now(tz='GB'),
        #             type='History',
        #             dataset=c,
        #         )
        #         obj.save()
        #     print("")

        if len(new_hist) > 0:
            if debug:
                print(new_hist)
            df_to_Model(new_hist, History, update=True)

        else:
            print("None")

        print("\nDataFrame information:")
        if len(hist) > 0:
            print("\nhist info:")
            print(hist.info())
            print("\nhist columns:", hist.columns.tolist())
        
        if len(new_hist) > 0:
            print("\nnew_hist info:")
            print(new_hist.info())
            print("\nnew_hist columns:", new_hist.columns.tolist())

        # First ensure each dataframe has unique indices
        if len(hist) > 0:
            hist = hist[~hist.index.duplicated(keep='last')]
        if len(new_hist) > 0:
            new_hist = new_hist[~new_hist.index.duplicated(keep='last')]

        # Ensure columns match
        if len(hist) > 0 and len(new_hist) > 0:
            # Get common columns
            common_cols = list(set(hist.columns) & set(new_hist.columns))
            hist = hist[common_cols]
            new_hist = new_hist[common_cols]
            print("\nCommon columns:", common_cols)

        # Now concatenate
        combined_hist = pd.concat([hist, new_hist])
        combined_hist = combined_hist[~combined_hist.index.duplicated(keep='last')].sort_index()
        hist = combined_hist

        if debug:
            print("Getting Historic Prices")

        prices, start = model_to_df(PriceHistory)
        
        # Get both outgoing and incoming rates from Octopus API
        agile_outgoing = get_agile(start=start, direction="outgoing")
        agile_incoming = get_agile(start=start, direction="incoming")
        
        # Convert outgoing rates to day ahead wholesale prices
        day_ahead = day_ahead_to_agile(agile_outgoing, reverse=True)
        
        new_prices = pd.concat([
            day_ahead.rename('day_ahead'), 
            agile_outgoing.rename('agile_outgoing'), 
            agile_incoming.rename('agile_incoming')
        ], axis=1)
        if len(prices) > 0:
            new_prices = new_prices[new_prices.index > prices.index[-1]]

        if debug:
            print(new_prices)

        if len(new_prices) > 0:
            print(new_prices)
            df_to_Model(new_prices, PriceHistory)
            prices = pd.concat([prices, new_prices]).sort_index()
            prices = prices[~prices.index.duplicated(keep='last')]  # Keep most recent in case of duplicates

        nordpool_data = get_nordpool(start=prices.index[-1] + pd.Timedelta("30min"))
        if nordpool_data is not None:
            nordpool = pd.DataFrame(nordpool_data).set_axis(["day_ahead"], axis=1)
            if len(nordpool) > 0:
                print(f"Hourly day ahead data used for period: {nordpool.index[0].strftime('%d-%b %H:%M')} - {nordpool.index[-1].strftime('%d-%b %H:%M')}")
                nordpool["agile"] = day_ahead_to_agile(nordpool["day_ahead"])
                prices = pd.concat([prices, nordpool]).sort_index()
        else:
            print("No Nordpool data available")

        if debug:
            print(f"Database prices:\n{prices}")
            print(f"New prices:\n{prices}")

        if drop_last > 0:
            print(f"drop_last: {drop_last}")
            print(f"len: {len(prices)} last:{prices.index[-1]}")
            prices = prices.iloc[:-drop_last]
            print(f"len: {len(prices)} last:{prices.index[-1]}")

        new_name = pd.Timestamp.now(tz="GB").strftime("%Y-%m-%d %H:%M")
        if new_name not in [f.name for f in Forecasts.objects.all()]:
            base_forecasts = Forecasts.objects.exclude(id__in=ignore_forecast).order_by("-created_at")
            last_forecasts ={forecast.created_at.date(): forecast.id for forecast in base_forecasts.order_by("created_at")}

            base_forecasts=base_forecasts.filter(id__in=[last_forecasts[k] for k in last_forecasts])

            if debug:
                print("Getting latest Forecast")
                print("Attempting to get forecast data...")
            fc, missing_fc = get_latest_forecast()
            if debug:
                print(f"Forecast data received. Rows: {len(fc) if fc is not None else 0}")
                print(f"Missing columns: {missing_fc}")

            if len(missing_fc) > 0:
                print(">>> ERROR: Unable to run forecast due to missing columns:", end="")
                for c in missing_fc:
                    print(c, end="")
                    obj=UpdateErrors(
                        date_time=pd.Timestamp.now(tz='GB'),
                        type='Forecast',
                        dataset=c,
                    )
                    obj.save()                

            else:
                if debug:
                    print(fc)

                if len(fc) > 0:
                    # Initialize prediction storage
                    outgoing_predictions = {}
                    incoming_predictions = {}
                    
                    for i in range(iters):
                        cols = hist.drop(drop_cols, axis=1).columns

                        X = hist[cols].iloc[-48 * np.random.randint(min_hist, max_hist) :]
                        y_outgoing = prices["agile_outgoing"].loc[X.index]  # Train directly on retail outgoing prices
                        y_incoming = prices["agile_incoming"].loc[X.index]  # Train directly on retail incoming prices

                        if not no_hist:
                            X1 = [X.copy()]
                            y1_outgoing = [y_outgoing.copy()]
                            y1_incoming = [y_incoming.copy()]
                        else:
                            X1 = []
                            y1_outgoing = []
                            y1_incoming = []

                        if not no_forecast:
                            for f in base_forecasts:
                                days_since_forecast = (pd.Timestamp.now(tz="GB") - f.created_at).days
                                if days_since_forecast < 28:
                                    df = get_forecast_from_model(forecast=f).loc[: prices.index[-1]]

                                    if len(df) > 0:
                                        rng = np.random.default_rng()
                                        max_len = DAYS_TO_INCLUDE * 48
                                        samples = rng.triangular(0, 0, max_len, int(max_len/2)).astype(int)
                                        samples = samples[samples < len(df)]
                                        if debug:
                                            print(
                                                f"{f.id:3d}:, {df.index[0].strftime('%d-%b %H:%M')} - {df.index[-1].strftime('%d-%b %H:%M')}  Length: {len(df.iloc[samples]):3d} Oversampling:{len(df.iloc[samples])/len(df) * 100:0.0f}% {len(samples)} {len(df.iloc[samples])}"
                                            )

                                        df = df.iloc[samples]

                                        X1.append(df[cols])
                                        y1_outgoing.append(prices["agile_outgoing"].loc[df.index])  # Train on retail outgoing prices
                                        y1_incoming.append(prices["agile_incoming"].loc[df.index])  # Train on retail incoming prices

                        X1 = pd.concat(X1)
                        y1_outgoing = pd.concat(y1_outgoing)
                        y1_incoming = pd.concat(y1_incoming)

                        # Remove any rows with NaN values in either target
                        valid_mask = ~(y1_outgoing.isna() | y1_incoming.isna())
                        X1 = X1[valid_mask]
                        y1_outgoing = y1_outgoing[valid_mask]
                        y1_incoming = y1_incoming[valid_mask]

                        # Train outgoing model
                        model_outgoing = xg.XGBRegressor(
                            objective="reg:squarederror",
                            booster="dart",
                            gamma=0.3,
                            eval_metric="rmse",
                        )

                        model_outgoing.fit(X1, y1_outgoing, verbose=True)
                        model_agile_outgoing = pd.Series(index=y1_outgoing.index, data=model_outgoing.predict(X1))

                        # Train incoming model
                        model_incoming = xg.XGBRegressor(
                            objective="reg:squarederror",
                            booster="dart",
                            gamma=0.3,
                            eval_metric="rmse",
                        )

                        model_incoming.fit(X1, y1_incoming, verbose=True)
                        model_agile_incoming = pd.Series(index=y1_incoming.index, data=model_incoming.predict(X1))

                        # Calculate errors - now comparing retail prices directly
                        rmse_outgoing = MSE(model_agile_outgoing, prices["agile_outgoing"].loc[X1.index]) ** 0.5
                        rmse_incoming = MSE(model_agile_incoming, prices["agile_incoming"].loc[X1.index]) ** 0.5

                        print(f"\nIteration: {i+1}", end="")
                        if debug:
                            print("\n--------------\n      ")
                        print(f" Outgoing RMS Error: {rmse_outgoing: 0.2f} p/kWh", end="")
                        print(f" Incoming RMS Error: {rmse_incoming: 0.2f} p/kWh", end="")
                        if debug:
                            print(f"\nLengths: History: {(len(X) / len(X1)*100):0.1f}%")
                            print(f"       Forecasts: {((len(X1) - len(X))/len(X1)*100):0.1f}%")
                        
                        # Store predictions for this iteration
                        outgoing_predictions[f"agile_outgoing_{i}"] = model_outgoing.predict(fc[cols])
                        incoming_predictions[f"agile_incoming_{i}"] = model_incoming.predict(fc[cols])

                    # After all iterations, create prediction DataFrames
                    outgoing_df = pd.DataFrame(outgoing_predictions, index=fc.index)
                    incoming_df = pd.DataFrame(incoming_predictions, index=fc.index)
                    
                    # Calculate statistics for outgoing predictions
                    outgoing_stats = pd.DataFrame({
                        'agile_outgoing': outgoing_df.mean(axis=1),
                        'agile_outgoing_low': outgoing_df.quantile(0.1, axis=1) if iters > 9 else outgoing_df.min(axis=1),
                        'agile_outgoing_high': outgoing_df.quantile(0.9, axis=1) if iters > 9 else outgoing_df.max(axis=1)
                    })
                    
                    # Ensure timezone consistency for outgoing stats
                    outgoing_stats.index = outgoing_stats.index.tz_convert("GB")
                    
                    # Calculate statistics for incoming predictions
                    incoming_stats = pd.DataFrame({
                        'agile_incoming': incoming_df.mean(axis=1),
                        'agile_incoming_low': incoming_df.quantile(0.1, axis=1) if iters > 9 else incoming_df.min(axis=1),
                        'agile_incoming_high': incoming_df.quantile(0.9, axis=1) if iters > 9 else incoming_df.max(axis=1)
                    })
                    
                    # Ensure timezone consistency for incoming stats
                    incoming_stats.index = incoming_stats.index.tz_convert("GB")
                    
                    # Combine all results
                    results = pd.concat([fc, outgoing_stats, incoming_stats], axis=1)
                    results.loc[:, 'wholesale'] = results['agile_outgoing']  # Using outgoing as reference

                    ag = pd.concat(
                        [
                            pd.DataFrame(
                                index=results.index,
                                data={
                                    "region": region,
                                    "agile_pred": results["agile_outgoing"].astype(float).round(2),
                                    "agile_low": results["agile_outgoing_low"].astype(float).round(2),
                                    "agile_high": results["agile_outgoing_high"].astype(float).round(2),
                                    "agile_incoming_pred": results["agile_incoming"].astype(float).round(2),
                                    "agile_incoming_low": results["agile_incoming_low"].astype(float).round(2),
                                    "agile_incoming_high": results["agile_incoming_high"].astype(float).round(2),
                                },
                            )
                            for region in regions
                        ]
                    )

                    # Clean up temporary columns and prepare final DataFrame
                    columns_to_keep = [col for col in results.columns if not (
                        col.startswith('agile_outgoing_') or 
                        col.startswith('agile_incoming_') or 
                        col in ["time", "day_of_week", "agile_outgoing_low", "agile_outgoing_high",
                               "agile_incoming_low", "agile_incoming_high"]
                    )]
                    fc = results[columns_to_keep].copy()  # Create a clean copy

                    this_forecast = Forecasts(name=new_name)
                    this_forecast.save()
                    fc.loc[:, "forecast"] = this_forecast
                    ag.loc[:, "forecast"] = this_forecast
                    df_to_Model(fc, ForecastData)
                    df_to_Model(ag, AgileData)

        if debug:
            for f in Forecasts.objects.all():
                print(f"{f.id:4d}: {f.name}")
        else:
            try:
                print(f"\n\nAdded Forecast: {this_forecast.id:>4d}: {this_forecast.name}")
            except:
                print("No forecast added")

        if reference_date:
            # Restore the original Timestamp.now function
            pd.Timestamp.now = original_now
