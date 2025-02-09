from django.db import models
from django.urls import reverse


class Forecasts(models.Model):
    name = models.CharField(unique=True, max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
    source = models.CharField(max_length=32, default='')  # No longer using manual/cron distinction

    def __str__(self):
        return self.name


class Nordpool(models.Model):
    date_time = models.DateTimeField(unique=True)
    day_ahead = models.FloatField()
    agile = models.FloatField()


class PriceHistory(models.Model):
    date_time = models.DateTimeField(unique=True)
    day_ahead = models.FloatField()
    agile = models.FloatField()
    agile_incoming = models.FloatField(default=0)  # Store incoming prices separately


class AgileData(models.Model):
    forecast = models.ForeignKey(Forecasts, related_name="prices", on_delete=models.CASCADE)
    region = models.CharField(max_length=1)
    agile_pred = models.FloatField()
    agile_low = models.FloatField()
    agile_high = models.FloatField()
    agile_incoming_pred = models.FloatField(default=0)
    agile_incoming_low = models.FloatField(default=0)
    agile_incoming_high = models.FloatField(default=0)
    date_time = models.DateTimeField()

    def get_absolute_url(self):
        return reverse("graph", kwargs={"region": self.region})


class UpdateErrors(models.Model):
    date_time = models.DateTimeField()
    type = models.CharField(max_length=10)
    dataset = models.CharField(max_length=32)


class History(models.Model):
    date_time = models.DateTimeField(unique=True)
    total_wind = models.FloatField()
    bm_wind = models.FloatField()
    solar = models.FloatField()
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()
    # other_gen_capacity = models.FloatField()
    # intercon_capacity = models.FloatField()


class ForecastData(models.Model):
    forecast = models.ForeignKey(Forecasts, related_name="data", on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    # Rename fields to clearly indicate they are retail prices
    agile_outgoing = models.FloatField()  # Renamed from day_ahead_outgoing
    agile_incoming = models.FloatField()  # Renamed from day_ahead_incoming
    wholesale = models.FloatField()  # Renamed from day_ahead
    bm_wind = models.FloatField()
    solar = models.FloatField()
    emb_wind = models.FloatField()
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()
    # other_gen_capacity = models.FloatField()
    # intercon_capacity = models.FloatField()

    # def __str__(self):
    #     return f"{self.date_time.strftime('%Y-%m-%dT%H:%M%Z') {self.agile:5.2f}}"
