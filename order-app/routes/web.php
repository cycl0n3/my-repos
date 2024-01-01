<?php

use App\Http\Controllers\ProfileController;
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\UserController;
use App\Http\Controllers\ProductController;
use App\Http\Controllers\BrandController;
use App\Http\Controllers\AdminController;

use Illuminate\Foundation\Application;
use Illuminate\Support\Facades\Route;
use Inertia\Inertia;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

Route::get('/', function () {
    return Inertia::render('Welcome', [
        'canLogin' => Route::has('login'),
        'canRegister' => Route::has('register'),
        'laravelVersion' => Application::VERSION,
        'phpVersion' => PHP_VERSION,
    ]);
});

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');

    Route::get('/dashboard', [DashboardController::class, 'dashboard'])->name('dashboard');

    Route::get('/admin/dashboard', [AdminController::class, 'admin_dashboard'])->name('admin.dashboard');
    Route::get('/admin/users/dashboard', [AdminController::class, 'admin_users_dashboard'])->name('admin.users.dashboard');
    Route::get('/admin/products/dashboard', [AdminController::class, 'admin_products_dashboard'])->name('admin.products.dashboard');
    Route::get('/admin/brands/dashboard', [AdminController::class, 'admin_brands_dashboard'])->name('admin.brands.dashboard');
    Route::get('/admin/brands/new', [AdminController::class, 'admin_brands_new'])->name('admin.brands.new');

    Route::get('/me', [UserController::class, 'me'])->name('me');
    Route::get('/user_count', [UserController::class, 'user_count'])->name('user_count');
    Route::get('/get_me', [UserController::class, 'get_me'])->name('get_me');
    Route::get('/brand_count', [BrandController::class, 'brand_count'])->name('brand_count');
    Route::get('/get_brands', [BrandController::class, 'get_brands'])->name('get_brands');

    Route::get('/product_count', [ProductController::class, 'product_count'])->name('product_count');
    Route::get('/get_products', [ProductController::class, 'get_products'])->name('get_products');

    // route to create brand
    Route::post('/admin/create_brand', [BrandController::class, 'create_brand'])->name('admin.create_brand');
});

require __DIR__.'/auth.php';
