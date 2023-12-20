<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

use App\Models\Brand;

class BrandController extends Controller
{
    // count number of brands
    public function brand_count(Request $request)
    {
        $brand_count = Brand::count();

        // sleep for 3 seconds to simulate a slow request
        // sleep(3);

        // return json response
        return response()->json([
            'brand_count' => $brand_count,
        ]);
    }

    // create new brand
    public function create_brand(Request $request)
    {
        Log::info('BrandController::create_brand()', [
            'name' => $request->name,
            'url' => $request->url,
            'description' => $request->description,
            'image' => $request->image,
        ]);

        try {
            // create new brand
            $brand = new Brand;

            $imageMimeType = $request->image->getClientMimeType();
            
            if (in_array($imageMimeType, ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'])) {
                $base64Image = 'data:' . $imageMimeType . ';base64,' . base64_encode(file_get_contents($request->image));

                $brand->name = $request->name;
                $brand->url = $request->url ?? '';
                $brand->description = $request->description;

                $brand->image = $base64Image;

                // sleep(3);

                $brand->save();
            } else {
                // invalid image
                return response()->json([
                    'message' => 'Invalid image',
                    'type' => 'error',
                ]);
            }

            // return json response
            return response()->json([
                'message' => 'Brand created successfully',
                'type' => 'success',
            ]);
        } catch (\Exception $e) {
            Log::error('BrandController::create_brand() - ' . $e->getMessage());

            return response()->json([
                'message' => 'Error creating brand',
                'type' => 'error',
            ]);
        }
    }
}
