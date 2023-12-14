<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Models\User;

use Inertia\Inertia;
use Inertia\Response;

class DashboardController extends Controller
{
  public function index(Request $request): Response
  {
    $user = $request->user();

    if ($user->roles == 'ADMIN') {
      $users = User::all();
      // load orders
      $users->load('orders');
      
      return Inertia::render('Dashboard', [
        'users' => $users,
      ]);
    } else {
      return Inertia::render('Dashboard', [
        'users' => [],
      ]);
    }
  }
}
