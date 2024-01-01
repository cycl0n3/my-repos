<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Models\User;
use App\Models\Product;
use App\Models\Brand;

use Inertia\Inertia;
use Inertia\Response;

class AdminController extends Controller
{
    public function admin_users_dashboard(Request $request): Response
	{
		$users = User::all();

		return Inertia::render('Admin/UsersDashboard', [
			'users' => $users,
		]);
	}

	public function admin_products_dashboard(Request $request): Response
	{
		$products = Product::all();

		return Inertia::render('Admin/ProductsDashboard', [
			'products' => $products,
		]);
	}

	public function admin_brands_dashboard(Request $request): Response
	{
		$brands = Brand::all();

		return Inertia::render('Admin/Brands/Dashboard', [
			'brands' => $brands,
		]);
	}

	public function admin_brands_new(Request $request): Response
	{
		return Inertia::render('Admin/Brands/New', [
		]);
	}
}
