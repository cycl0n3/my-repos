<?php

use App\Http\Controllers\ProfileController;
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\UserController;
use App\Http\Controllers\ProductController;
use App\Http\Controllers\BrandController;

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
    Route::get('/dashboard/admin/users', [DashboardController::class, 'dashboard_admin_users'])->name('dashboard.admin.users');
    Route::get('/dashboard/admin/products', [DashboardController::class, 'dashboard_admin_products'])->name('dashboard.admin.products');
    Route::get('/dashboard/admin/brands', [DashboardController::class, 'dashboard_admin_brands'])->name('dashboard.admin.brands');

    Route::get('/me', [UserController::class, 'me']);
    Route::get('/user_count', [UserController::class, 'user_count']);
    Route::get('/get_me', [UserController::class, 'get_me']);
    Route::get('/brand_count', [BrandController::class, 'brand_count']);   

    Route::get('/product_count', [ProductController::class, 'product_count']);
});

require __DIR__.'/auth.php';
