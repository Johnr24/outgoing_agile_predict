# Generated by Django 4.2.11 on 2025-02-09 15:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0002_pricehistory_agile_incoming"),
    ]

    operations = [
        migrations.AddField(
            model_name="forecastdata",
            name="day_ahead_high",
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name="forecastdata",
            name="day_ahead_low",
            field=models.FloatField(default=0),
        ),
    ]
