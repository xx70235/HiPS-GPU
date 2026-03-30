/**
 * WCSLIB坐标转换测试程序
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include "fits_io.h"
#include "wcslib_transform.h"
#include "coordinate_transform.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <fits_file>" << std::endl;
        return 1;
    }
    
    std::string fitsFile = argv[1];
    
    FitsData fitsData = FitsReader::readFitsFile(fitsFile);
    if (fitsData.width == 0 || fitsData.height == 0) {
        std::cerr << "Failed to read FITS file: " << fitsFile << std::endl;
        return 1;
    }
    
    std::cout << "FITS file: " << fitsFile << std::endl;
    std::cout << "Image size: " << fitsData.width << " x " << fitsData.height << std::endl;
    std::cout << "CRPIX: (" << fitsData.crpix1 << ", " << fitsData.crpix2 << ")" << std::endl;
    std::cout << "CRVAL: (" << fitsData.crval1 << ", " << fitsData.crval2 << ")" << std::endl;
    std::cout << "CTYPE: (" << fitsData.ctype1 << ", " << fitsData.ctype2 << ")" << std::endl;
    std::cout << "CD matrix: " << std::endl;
    std::cout << "  [" << fitsData.cd1_1 << ", " << fitsData.cd1_2 << "]" << std::endl;
    std::cout << "  [" << fitsData.cd2_1 << ", " << fitsData.cd2_2 << "]" << std::endl;
    std::cout << std::endl;
    
    WCSLibTransform wcslib(fitsData);
    CoordinateTransform original(fitsData);
    
    if (!wcslib.isValid()) {
        std::cerr << "Failed to initialize WCSLIB transform" << std::endl;
        return 1;
    }
    
    std::cout << "=== Pixel to World Test ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    std::vector<std::pair<double, double>> testPixels = {
        {0.0, 0.0},
        {312.0, 312.0},
        {624.0, 624.0},
        {156.0, 156.0},
        {468.0, 468.0}
    };
    
    for (const auto& px : testPixels) {
        double ra_wcs, dec_wcs;
        int status = wcslib.pixelToWorld(px.first, px.second, ra_wcs, dec_wcs);
        CelestialCoord orig = original.pixelToCelestial(Coord(px.first, px.second));
        
        std::cout << "Pixel (" << px.first << ", " << px.second << "):" << std::endl;
        std::cout << "  WCSLIB:   RA=" << ra_wcs << ", Dec=" << dec_wcs;
        if (status) std::cout << " [status=" << status << "]";
        std::cout << std::endl;
        std::cout << "  Original: RA=" << orig.ra << ", Dec=" << orig.dec << std::endl;
        std::cout << "  Diff:     dRA=" << (ra_wcs - orig.ra) * 3600 << " arcsec, dDec=" 
                  << (dec_wcs - orig.dec) * 3600 << " arcsec" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "=== World to Pixel Test ===" << std::endl;
    
    std::vector<std::pair<double, double>> testWorld = {
        {fitsData.crval1, fitsData.crval2},
        {fitsData.crval1 + 1.0, fitsData.crval2},
        {fitsData.crval1 - 1.0, fitsData.crval2},
        {fitsData.crval1, fitsData.crval2 + 1.0},
        {fitsData.crval1, fitsData.crval2 - 1.0}
    };
    
    for (const auto& wc : testWorld) {
        double x_wcs, y_wcs;
        int status = wcslib.worldToPixel(wc.first, wc.second, x_wcs, y_wcs);
        Coord orig = original.celestialToPixel(CelestialCoord(wc.first, wc.second));
        
        std::cout << "World (" << wc.first << ", " << wc.second << "):" << std::endl;
        std::cout << "  WCSLIB:   x=" << x_wcs << ", y=" << y_wcs;
        if (status) std::cout << " [status=" << status << "]";
        std::cout << std::endl;
        std::cout << "  Original: x=" << orig.x << ", y=" << orig.y << std::endl;
        std::cout << "  Diff:     dx=" << (x_wcs - orig.x) << " px, dy=" << (y_wcs - orig.y) << " px" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "=== Round-trip Test (Pixel -> World -> Pixel) ===" << std::endl;
    
    double maxError = 0.0;
    int numTests = 0;
    
    for (int y = 0; y < fitsData.height; y += 100) {
        for (int x = 0; x < fitsData.width; x += 100) {
            double ra, dec;
            int status1 = wcslib.pixelToWorld(x, y, ra, dec);
            if (status1) continue;
            
            double x2, y2;
            int status2 = wcslib.worldToPixel(ra, dec, x2, y2);
            if (status2) continue;
            
            double error = sqrt((x2 - x) * (x2 - x) + (y2 - y) * (y2 - y));
            if (error > maxError) maxError = error;
            numTests++;
        }
    }
    
    std::cout << "Tested " << numTests << " points" << std::endl;
    std::cout << "Max round-trip error: " << maxError << " pixels" << std::endl;
    
    return 0;
}
