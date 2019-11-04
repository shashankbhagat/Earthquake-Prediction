import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import EarthQuake_phase_1 as quake
import cartopy.crs as ccrs
from cmocean import cm as cmo
from cartopy.util import add_cyclic_point
import cartopy
from cartopy.feature import NaturalEarthFeature
from matplotlib.cm import get_cmap
import matplotlib


obj=quake.EarthQuakePhase1()
Latitude,Longitude,Magnitude=obj.loadData_Graph()
#Latitude=Latitude[0:1000]
#Longitude=Longitude[0:1000]
#Magnitude=Magnitude[0:1000]
fig3D=plt.figure()

#Scatter plot
ax3D=fig3D.add_subplot(111,projection='3d')

ax3D.scatter(Latitude[:5000],Longitude[:5000],Magnitude[:5000],color='#800080',marker='o')

ax3D.set_xlabel('Latitude')
ax3D.set_ylabel('Longitude')
ax3D.set_zlabel('Magnitude')

plt.show()

axMap=plt.axes(projection=ccrs.PlateCarree())
axMap.coastlines()

#axMap.add_feature(cartopy.feature.LAND)
#axMap.add_feature(cartopy.feature.OCEAN)
#axMap.add_feature(cartopy.feature.COASTLINE)
#axMap.add_feature(cartopy.feature.BORDERS,linestyle=':')
#axMap.add_feature(cartopy.feature.LAKES,alpha=0.5)
#axMap.add_feature(cartopy.feature.RIVERS)

#axMap.stock_img()

plt.title("Eathquake plot")


#USA lat lon range
#axMap.set_extent([-130,-55.09,52.66,12.75])

#for lat,lon in zip(Latitude,Longitude):
#    plt.plot(lon,lat, marker='o' ,transform=ccrs.PlateCarree(),markersize=5, color="r")

cmap=get_cmap('cool')
normalize=matplotlib.colors.Normalize(vmin=min(Magnitude),vmax=max(Magnitude))
colors=[cmap(normalize(value)) for value in Magnitude]

axMap.scatter(Longitude,Latitude,transform=ccrs.PlateCarree(),color=colors)

cax, _ = matplotlib.colorbar.make_axes(axMap)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

plt.show()




