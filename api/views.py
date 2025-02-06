from django.shortcuts import render
import pandas as pd


# Create your views here.
from rest_framework import generics
from prices.models import Forecasts
from .serializers import PriceForecastSerializer, PriceForecastRegionSerializer


class PriceForecastAPIView(generics.ListAPIView):
    serializer_class = PriceForecastSerializer

    def get_queryset(self):
        ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:3]]
        return Forecasts.objects.filter(id__in=ids)


class PriceForecastRegionAPIView(generics.ListAPIView):
    serializer_class = PriceForecastRegionSerializer

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        context.update({"region": self.kwargs["region"].upper()})
        return context

    def get_queryset(self):
        #     """
        #     This view should return a list of all the purchases for
        #     the user as determined by the username portion of the URL.
        #     """
        #     update_if_required()

        ids = [f.id for f in Forecasts.objects.all().order_by("-created_at")[:3]]

        queryset = Forecasts.objects.filter(id__in=ids).order_by("-created_at")
        return queryset
