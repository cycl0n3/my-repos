<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use Inertia\Inertia;
use Inertia\Response;

use Illuminate\Http\RedirectResponse;

class DashboardController extends Controller
{
	public function dashboard(Request $request): Response | RedirectResponse
	{
		$user = $request->user();

		if ($user->roles == 'ADMIN') {
			// redirect o admin dashboard
			return redirect(route('admin.dashboard'));
		} else {
			return Inertia::render('User/Dashboard', [
			]);
		}
	}
}
