import "@/Pages/Dashboard.css";

import { useState, useEffect } from "react";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";
import AdminBlock from "@/Layouts/AdminBlock";

import { Head } from "@inertiajs/react";

import manager from "../../images/the-manager.svg";
import worker from "../../images/the-worker.svg";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

function getOrders() {
    return fetch(`/orders`)
        .then((res) => res.json())
}

export default function Dashboard({ auth, users }) {
    const queryClient = useQueryClient();

    const orderQuery = useQuery({
        queryKey: ["orders"],
        queryFn: getOrders,
    });

    return (
        <AuthenticatedLayout
            user={auth.user}
            header={
                <h2 className="font-semibold text-xl text-gray-800 dark:text-gray-200 leading-tight">
                    <p className="neon--heading">Dashboard</p>
                </h2>
            }
        >
            <Head title="Dashboard" />

            <div className="py-12">
                <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
                    <div className="bg-white dark:bg-gray-800 overflow-hidden shadow-sm sm:rounded-lg">
                        <div className="p-6 text-gray-900 dark:text-gray-100">
                            <div className="card">
                                <table className="min-w-full divide-y divide-gray-200">
                                    <thead className="bg-gray-50 dark:bg-gray-700">
                                        <tr>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Name
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Email
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Orders
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Roles
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-dark divide-y divide-gray-200">
                                        <tr>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="neon--purple">
                                                    {auth.user.name}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="neon--blue">
                                                    {auth.user.email}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                {orderQuery.data ? (
                                                    JSON.stringify(orderQuery.data.orders)
                                                ) : (
                                                    <div className="neon--red">
                                                        Loading...
                                                    </div>
                                                )}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="neon--red">
                                                    <img
                                                        src={
                                                            auth.user.roles ===
                                                            "ADMIN"
                                                                ? manager
                                                                : worker
                                                        }
                                                        alt="roles"
                                                        className="w-12 h-12 inline-block"
                                                    />
                                                </div>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <AdminBlock user={auth.user}>
                <div className="py-12">
                    <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
                        <div className="bg-white dark:bg-gray-800 overflow-hidden shadow-sm sm:rounded-lg">
                            <div className="p-6 text-gray-900 dark:text-gray-100">
                                <table className="min-w-full divide-y divide-gray-200">
                                    <thead className="bg-gray-50 dark:bg-gray-700">
                                        <tr>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Name
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Email
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Roles
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Orders
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                                Actions
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-dark divide-y divide-gray-200">
                                        {users.map((user) => (
                                            <tr key={user.id}>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <div className="neon--purple">
                                                        {user.name}
                                                    </div>
                                                </td>
                                                <td
                                                    className={`px-6 py-4 whitespace-nowrap text-blue-500`}
                                                >
                                                    <a
                                                        href={`users/info/${user.id}`}
                                                        className="neon--blue"
                                                    >
                                                        {user.email}
                                                    </a>
                                                </td>
                                                <td
                                                    className={`px-6 py-4 whitespace-nowrap ${
                                                        user.roles === "ADMIN"
                                                            ? "text-red-500"
                                                            : "text-green-500"
                                                    }`}
                                                >
                                                    {/* {user.roles} */}
                                                    <img
                                                        src={
                                                            user.roles ===
                                                            "ADMIN"
                                                                ? manager
                                                                : worker
                                                        }
                                                        alt="roles"
                                                        className="w-12 h-12 inline-block"
                                                    />
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    {user.orders.length}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <div className="neon--red">
                                                        <a
                                                            href={`users/edit/${user.id}`}
                                                        >
                                                            edit
                                                        </a>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </AdminBlock>
        </AuthenticatedLayout>
    );
}
