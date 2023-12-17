import "@/Pages/Dashboard.css";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";

import { Head } from "@inertiajs/react";

import manager from "../../../images/the-manager.svg";

export default function Dashboard({ auth }) {
    const totalUsers = 100; // Replace with actual total users count
    const totalProducts = 50; // Replace with actual total products count

    return (
        <AuthenticatedLayout
            user={auth.user}
            header={
                <h2 className="font-semibold text-xl text-gray-800 dark:text-gray-200 leading-tight">
                    <p className="neon--heading">Admin Dashboard</p>
                </h2>
            }
        >
            <Head title="Admin Dashboard" />

            <div className="py-12">
                <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
                    <div className="bg-white dark:bg-gray-800 overflow-hidden shadow-sm sm:rounded-lg">
                        <div className="p-6 text-gray-900 dark:text-gray-100">
                            <div className="card">
                                <div className="min-w-full max-w-sm bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
                                    <div className="flex flex-col items-center pb-10">
                                        <img
                                            className="w-24 h-24 mb-3 rounded-full shadow-lg"
                                            src={manager}
                                            alt="Manager image"
                                        />
                                        <h5 className="mb-1 text-xl font-medium neon--purple">
                                            {auth.user.name}
                                        </h5>
                                        <span className="text-sm neon--blue">
                                            {auth.user.email}
                                        </span>
                                        <div className="flex mt-4 md:mt-6">
                                            <div className="mr-6">
                                                <label className="text-gray-500 dark:text-gray-300">Total Users:</label>
                                                <div className="text-2xl font-semibold neon--purple">{totalUsers}</div>
                                            </div>
                                            <div>
                                                <label className="text-gray-500 dark:text-gray-300">Total Products:</label>
                                                <div className="text-2xl font-semibold neon--purple">{totalProducts}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </AuthenticatedLayout>
    );
}
