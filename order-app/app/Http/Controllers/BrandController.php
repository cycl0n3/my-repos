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

        // return json response
        return response()->json([
            'brand_count' => $brand_count,
        ]);
    }
}
