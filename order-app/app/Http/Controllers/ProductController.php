<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

use App\Models\Product;

class ProductController extends Controller
{
    public function product_count(Request $request)
    {
        $product_count = DB::table('products')->count();

        // return json response
        return response()->json([
            'product_count' => $product_count,
        ]);
    }
}
