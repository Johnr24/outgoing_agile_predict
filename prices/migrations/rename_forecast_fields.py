from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('prices', '0006_alter_pricehistory_agile_incoming'),
    ]

    operations = [
        migrations.RenameField(
            model_name='forecastdata',
            old_name='day_ahead_outgoing',
            new_name='agile_outgoing',
        ),
        migrations.RenameField(
            model_name='forecastdata',
            old_name='day_ahead_incoming',
            new_name='agile_incoming',
        ),
        migrations.RenameField(
            model_name='forecastdata',
            old_name='day_ahead',
            new_name='wholesale',
        ),
    ] 