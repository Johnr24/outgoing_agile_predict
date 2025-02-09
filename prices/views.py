import pandas as pd

# Create your views here.
from django.views.generic import TemplateView, FormView
from .models import Forecasts, PriceHistory, AgileData, ForecastData, History, UpdateErrors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile
from .forms import ForecastForm


regions = GLOBAL_SETTINGS["REGIONS"]
PRIOR_DAYS = 2


class GlossaryView(TemplateView):
    template_name = "base.html"


class ColorView(TemplateView):
    template_name = "color_mode.html"


class ApiHowToView(TemplateView):
    template_name = "api_how_to.html"


class AboutView(TemplateView):
    template_name = "about.html"


class HomeAssistantView(TemplateView):
    template_name = "home_assistant.html"


class StatsView(TemplateView):
    template_name = "stats.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        agile_actuals_end = pd.Timestamp(PriceHistory.objects.all().order_by("-date_time")[0].date_time)
        agile_actuals_start = agile_actuals_end - pd.Timedelta("7D")

        agile_actuals_objects = PriceHistory.objects.filter(date_time__gt=agile_actuals_start).order_by("date_time")
        df = pd.DataFrame(
            index=[obj.date_time for obj in agile_actuals_objects],
            data={
                "actuals_outgoing": [obj.agile for obj in agile_actuals_objects],
                "actuals_incoming": [obj.agile * 2 for obj in agile_actuals_objects],  # Approximate for display
            },
        )

        agile_forecast_data = AgileData.objects.filter(
            date_time__gt=agile_actuals_start, date_time__lte=agile_actuals_end
        )
        figure = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=("Outgoing Agile Price", "Outgoing Error HeatMap", 
                          "Incoming Agile Price", "Incoming Error HeatMap"),
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )

        # Process predictions
        for forecast in agile_forecast_data.values_list("forecast").distinct():
            forecast_created_at = pd.Timestamp(Forecasts.objects.filter(id=forecast[0])[0].created_at).tz_convert("GB")
            forecast_after = (
                pd.Timestamp.combine(forecast_created_at.date(), pd.Timestamp("22:00").time())
                .tz_localize("UTC")
                .tz_convert("GB")
            )

            if forecast_created_at.hour >= 16:
                forecast_after += pd.Timedelta("24h")

            agile_pred_objects = agile_forecast_data.filter(forecast=forecast[0])
            index = [
                obj.date_time
                for obj in agile_pred_objects
                if obj.date_time > forecast_after
                if obj.date_time < forecast_after + pd.Timedelta("7D")
            ]
            outgoing_data = [
                obj.agile_pred
                for obj in agile_pred_objects
                if obj.date_time > forecast_after
                if obj.date_time < forecast_after + pd.Timedelta("7D")
            ]
            incoming_data = [
                obj.agile_incoming_pred
                for obj in agile_pred_objects
                if obj.date_time > forecast_after
                if obj.date_time < forecast_after + pd.Timedelta("7D")
            ]
            
            if len(outgoing_data) > 0:
                df.loc[index, f"outgoing_{forecast_created_at}"] = outgoing_data
                df.loc[index, f"incoming_{forecast_created_at}"] = incoming_data
                
                # Plot outgoing prediction
                figure.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[f"outgoing_{forecast_created_at}"],
                        line={"color": "grey", "width": 0.5},
                        showlegend=False,
                        mode="lines",
                    ),
                    row=1,
                    col=1,
                )
                
                # Plot incoming prediction
                figure.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[f"incoming_{forecast_created_at}"],
                        line={"color": "grey", "width": 0.5},
                        showlegend=False,
                        mode="lines",
                    ),
                    row=3,
                    col=1,
                )

        # Add actual values
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["actuals_outgoing"],
                line={"color": "yellow", "width": 3},
                name="Actual Outgoing",
            ),
            row=1,
            col=1,
        )
        
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["actuals_incoming"],
                line={"color": "yellow", "width": 3},
                name="Actual Incoming",
            ),
            row=3,
            col=1,
        )

        # Create heatmaps for both outgoing and incoming
        outgoing_cols = [col for col in df.columns if col.startswith('outgoing_') and col != 'actuals_outgoing']
        incoming_cols = [col for col in df.columns if col.startswith('incoming_') and col != 'actuals_incoming']
        
        # Process outgoing errors
        df_outgoing = df.copy()
        for x in outgoing_cols:
            df_outgoing[x] = abs(df_outgoing[x] - df_outgoing["actuals_outgoing"])
        df_outgoing_plot = df_outgoing[outgoing_cols].sort_index(axis=1).T
        df_outgoing_plot = df_outgoing_plot.loc[df_outgoing_plot.index > agile_actuals_start - pd.Timedelta("3D")]
        
        # Process incoming errors
        df_incoming = df.copy()
        for x in incoming_cols:
            df_incoming[x] = abs(df_incoming[x] - df_incoming["actuals_incoming"])
        df_incoming_plot = df_incoming[incoming_cols].sort_index(axis=1).T
        df_incoming_plot = df_incoming_plot.loc[df_incoming_plot.index > agile_actuals_start - pd.Timedelta("3D")]

        # Add outgoing heatmap
        figure.add_heatmap(
            x=df_outgoing_plot.columns,
            y=df_outgoing_plot.index,
            z=df_outgoing_plot.to_numpy(),
            row=2,
            col=1,
            colorbar={"title": "Outgoing Error\n[p/kWh]", "x": 1.02, "y": 0.75}
        )
        
        # Add incoming heatmap
        figure.add_heatmap(
            x=df_incoming_plot.columns,
            y=df_incoming_plot.index,
            z=df_incoming_plot.to_numpy(),
            row=4,
            col=1,
            colorbar={"title": "Incoming Error\n[p/kWh]", "x": 1.02, "y": 0.25}
        )

        layout = dict(
            height=1200,  # Increased height for 4 subplots
            template="plotly_dark",
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
            margin={"r": 70, "t": 50},  # Increased right margin for colorbar titles
        )

        figure.update_layout(**layout)
        figure.update_yaxes(title_text="Outgoing Price [p/kWh]", row=1, col=1)
        figure.update_yaxes(title_text="Incoming Price [p/kWh]", row=3, col=1)

        context["stats"] = figure.to_html()
        return context


class GraphFormView(FormView):
    form_class = ForecastForm
    template_name = "graph.html"

    def get_form_kwargs(self):
        kwargs = super(GraphFormView, self).get_form_kwargs()
        kwargs["prefix"] = "test"
        return kwargs

    def update_chart(self, context, **kwargs):
        region = context["region"]
        forecasts_to_plot = kwargs.get("forecasts_to_plot")
        days_to_plot = int(kwargs.get("days_to_plot", 14))
        show_generation_and_demand = kwargs.get("show_generation_and_demand", True)
        show_range = kwargs.get("show_range_on_most_recent_forecast", True)
        show_overlap = kwargs.get("show_forecast_overlap", False)

        first_forecast = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]
        first_forecast_data = ForecastData.objects.filter(forecast=first_forecast).order_by("date_time")
        forecast_start = first_forecast_data[0].date_time
        if len(first_forecast_data) >= 48 * days_to_plot:
            forecast_end = first_forecast_data[48 * days_to_plot].date_time
        else:
            forecast_end = [d.date_time for d in first_forecast_data][-1]

        price_start = PriceHistory.objects.all().order_by("-date_time")[48 * PRIOR_DAYS].date_time
        start = min(forecast_start, price_start)

        figure = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Agile Outgoing", "Agile Incoming", "Generation and Demand"),
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        # Get historical prices
        p = PriceHistory.objects.filter(date_time__gte=start).order_by("-date_time")
        day_ahead = pd.Series(index=[a.date_time for a in p], data=[a.day_ahead for a in p])
        agile = day_ahead_to_agile(day_ahead, region=region).sort_index()

        # Add the actual price trace to both plots
        figure.add_trace(
            go.Scatter(
                x=agile.loc[:forecast_end].index.tz_convert("GB"),
                y=agile.loc[:forecast_end],
                marker={"symbol": 104, "size": 1, "color": "white"},
                mode="lines",
                name="Actual Outgoing",
            ),
            row=1,
            col=1,
        )

        limit = None
        width = 3
        for f in Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at"):
            d = AgileData.objects.filter(forecast=f, region=region)
            if len(d) > 0:
                if limit is None:
                    d = d[: (48 * days_to_plot)]
                    limit = d[-1].date_time
                else:
                    d = list(d.filter(date_time__lte=limit))

                x = [a.date_time for a in d if (a.date_time >= agile.index[-1] or show_overlap)]
                y = [a.agile_pred for a in d if (a.date_time >= agile.index[-1] or show_overlap)]
                y_incoming = [a.agile_incoming_pred for a in d if (a.date_time >= agile.index[-1] or show_overlap)]

                df = pd.Series(index=pd.to_datetime(x), data=y).sort_index()
                df.index = df.index.tz_convert("GB")
                df = df.loc[agile.index[0] :]

                df_incoming = pd.Series(index=pd.to_datetime(x), data=y_incoming).sort_index()
                df_incoming.index = df_incoming.index.tz_convert("GB")
                df_incoming = df_incoming.loc[agile.index[0] :]

                # Plot outgoing prediction
                figure.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df,
                        marker={"symbol": 104, "size": 10},
                        mode="lines",
                        line=dict(width=width),
                        name=f"Outgoing {f.name}",
                    ),
                    row=1,
                    col=1,
                )

                # Plot incoming prediction
                figure.add_trace(
                    go.Scatter(
                        x=df_incoming.index,
                        y=df_incoming,
                        marker={"symbol": 104, "size": 10},
                        mode="lines",
                        line=dict(width=width),
                        name=f"Incoming {f.name}",
                    ),
                    row=2,
                    col=1,
                )

                if (width == 3) and (d[0].agile_high != d[0].agile_low and show_range):
                    # Add outgoing range
                    figure.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=[a.agile_low for a in d if (a.date_time >= agile.index[-1] or show_overlap)],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="red"),
                            name="Outgoing Low",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=[a.agile_high for a in d if (a.date_time >= agile.index[-1] or show_overlap)],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="red"),
                            name="Outgoing High",
                            showlegend=False,
                            fill="tonexty",
                            fillcolor="rgba(255,127,127,0.5)",
                        ),
                        row=1,
                        col=1,
                    )

                    # Add incoming range
                    figure.add_trace(
                        go.Scatter(
                            x=df_incoming.index,
                            y=[a.agile_incoming_low for a in d if (a.date_time >= agile.index[-1] or show_overlap)],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="blue"),
                            name="Incoming Low",
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=df_incoming.index,
                            y=[a.agile_incoming_high for a in d if (a.date_time >= agile.index[-1] or show_overlap)],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="blue"),
                            name="Incoming High",
                            showlegend=False,
                            fill="tonexty",
                            fillcolor="rgba(127,127,255,0.5)",
                        ),
                        row=2,
                        col=1,
                    )
                width = 1

        if show_generation_and_demand:
            # Add generation and demand traces
            f = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]
            d = ForecastData.objects.filter(forecast=f, date_time__lte=forecast_end).order_by("date_time")
            
            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in d],
                    y=[a.demand / 1000 for a in d],
                    line={"color": "cyan", "width": 3},
                    name="Forecast National Demand",
                ),
                row=3,
                col=1,
            )

            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in d],
                    y=[a.bm_wind / 1000 for a in d],
                    fill="tozeroy",
                    line={"color": "rgba(63,127,63)"},
                    fillcolor="rgba(127,255,127,0.8)",
                    name="Forecast Metered Wind",
                ),
                row=3,
                col=1,
            )

            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in d],
                    y=[(a.emb_wind + a.bm_wind) / 1000 for a in d],
                    fill="tonexty",
                    line={"color": "blue", "width": 1},
                    fillcolor="rgba(127,127,255,0.8)",
                    name="Forecast Embedded Wind",
                ),
                row=3,
                col=1,
            )

            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in d],
                    y=[(a.solar + a.emb_wind + a.bm_wind) / 1000 for a in d],
                    fill="tonexty",
                    line={"color": "lightgray", "width": 3},
                    fillcolor="rgba(255,255,127,0.8)",
                    name="Forecast Solar",
                ),
                row=3,
                col=1,
            )

            h = History.objects.filter(date_time__gte=start, date_time__lte=forecast_end)

            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in h],
                    y=[a.demand / 1000 for a in h],
                    line={"color": "#aaaa77", "width": 2},
                    name="Historic Demand",
                ),
                row=3,
                col=1,
            )

            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in h],
                    y=[(a.total_wind + a.solar) / 1000 for a in h],
                    line={"color": "red", "width": 2},
                    name="Historic Solar + Wind",
                ),
                row=3,
                col=1,
            )

        layout = dict(
            yaxis={"title": "Outgoing Agile Price [p/kWh]"},
            yaxis2={"title": "Incoming Agile Price [p/kWh]"},
            yaxis3={"title": "Power [GW]" if show_generation_and_demand else ""},
            margin={
                "r": 5,
                "t": 50,
            },
            height=800,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="right", x=1),
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
        )

        figure.update_layout(**layout)
        figure.update_yaxes(
            title_text="Outgoing Agile Price [p/kWh]",
            row=1,
            col=1,
            fixedrange=True,
        )
        figure.update_yaxes(
            title_text="Incoming Agile Price [p/kWh]",
            row=2,
            col=1,
            fixedrange=True,
        )
        figure.update_yaxes(
            title_text="Power [GW]" if show_generation_and_demand else "",
            row=3,
            col=1,
            fixedrange=True,
        )
        figure.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86000000], value="%H:%M\n%a %d %b"),
                dict(dtickrange=[86000000, None], value="%a\n%b %d"),
            ],
        )

        context["graph"] = figure.to_html(
            config={
                "modeBarButtonsToRemove": [
                    "zoom",
                    "pan",
                    "select",
                    "zoomIn",
                    "zoomOut",
                    "autoScale",
                    "resetScale",
                ]
            }
        )
        for error_type in ["history", "forecast"]:
            context[f"{error_type}_errors"] = [
                {
                    "date_time": pd.Timestamp(x.date_time).tz_convert("GB"),
                    "dataset": GLOBAL_SETTINGS["DATASETS"][x.dataset]["name"],
                    "source": GLOBAL_SETTINGS["DATASETS"][x.dataset]["source"],
                }
                for x in list(UpdateErrors.objects.filter(type=error_type.title()))
            ]
        return context

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            f = Forecasts.objects.latest("created_at")
            region = self.kwargs.get("region", "X").upper()
            context["region"] = region
            context["region_name"] = regions.get(region, {"name": ""})["name"]
            context = self.update_chart(context=context, forecasts_to_plot=[f.id])
        except Forecasts.DoesNotExist:
            # Handle case when no forecasts exist
            region = self.kwargs.get("region", "X").upper()
            context["region"] = region
            context["region_name"] = regions.get(region, {"name": ""})["name"]
            context["error_message"] = "No forecasts available yet. Please wait for the next update."
        return context

    def form_valid(self, form):
        context = self.get_context_data(form=form)
        context = self.update_chart(context=context, **form.cleaned_data)
        return self.render_to_response(context=context)

    def form2_valid(self, form):
        context = self.get_context_data(form=form)
        context = self.update_chart(context=context, **form.cleaned_data)
        return self.render_to_response(context=context)
