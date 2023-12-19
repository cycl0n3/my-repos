<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

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
        // validate request
        $request->validate([
            'name' => 'required|unique:brands',
        ]);

        // create new brand
        $brand = new Brand;
        
        $brand->name = $request->name;
        $brand->description = $request->description;


        $brand->save();

        // return json response
        return response()->json([
            'message' => 'Brand created successfully',
        ]);
    }
}
