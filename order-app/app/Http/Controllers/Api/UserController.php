<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;

class UserController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        // return 404 code
        return response()->json([
            'message' => 'Not Found',
        ], 404);
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        // return 404 code
        return response()->json([
            'message' => 'Not Found',
        ], 404);
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        // return 404 code
        return response()->json([
            'message' => 'Not Found',
        ], 404);
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        // return 404 code
        return response()->json([
            'message' => 'Not Found',
        ], 404);
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        // return 404 code
        return response()->json([
            'message' => 'Not Found',
        ], 404);
    }

    // get number of orders for current user using pure sql
    public function orders(Request $request)
    {
        // $user = $request->user();

        // $orders = \DB::select('SELECT COUNT(*) AS orders FROM orders WHERE user_id = ?', [$user->id]);

        return response()->json([
            'orders' => 0
        ]);

    }
}
