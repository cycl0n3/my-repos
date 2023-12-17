<?php

namespace App\Http\Controllers;

use App\Http\Controllers\Controller;

use Illuminate\Http\Request;

use Illuminate\Support\Facades\DB;

class UserController extends Controller
{
    public function orders(Request $request)
    {
        $user = $request->user();
    
        $orders = DB::table('orders')
        ->join('products', 'orders.product_id', '=', 'products.id')
        ->select('orders.*', 'products.name as product_name', 'products.price as product_price')
        ->where('orders.user_id', '=', $user->id)
        ->get();
    
        return view('orders', [
            'orders' => $orders,
        ]);
    }

    // User count
    public function user_count(Request $request)
    {
        $user = $request->user();

        $user_count = DB::table('users')->count();

        return view('user_count', [
            'user_count' => $user_count,
        ]);
    }

    // Get me
    public function get_me(Request $request)
    {
        $user = $request->user();

        return view('get_me', [
            'user' => $user,
        ]);
    }
}
