<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Models\User;

use Inertia\Inertia;
use Inertia\Response;

class DashboardController extends Controller
{
  public function dashboard(Request $request): Response
  {
    $user = $request->user();

    if ($user->roles == 'ADMIN') {
      return Inertia::render('Admin/Dashboard', [
      ]);
    } else {
      return Inertia::render('User/Dashboard', [
      ]);
    }
  }

  public function dashboard_admin_users(Request $request): Response
  {
    $users = User::all();

    return Inertia::render('Admin/UsersDashboard', [
      'users' => $users,
    ]);
  }
}
