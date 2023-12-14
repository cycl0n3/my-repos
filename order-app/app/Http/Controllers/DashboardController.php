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
      return Inertia::render('Dashboard', [
        'users' => User::all(),
      ]);
    } else {
      return Inertia::render('Dashboard', [
        'users' => [],
      ]);
    }
  }
}
