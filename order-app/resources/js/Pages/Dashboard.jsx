import "@/Pages/Dashboard.css";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";
import AdminLayout from "@/Layouts/AdminLayout";

import { Head } from "@inertiajs/react";

export default function Dashboard({ auth, users }) {
    console.log("Dashboard", users);

    return (
        <AuthenticatedLayout
            user={auth.user}
            header={
                <h2 className="font-semibold text-xl text-gray-800 dark:text-gray-200 leading-tight">
                    Dashboard
                </h2>
            }
        >
            <Head title="Dashboard" />

            <div className="py-12">
                <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
                    <div className="bg-white dark:bg-gray-800 overflow-hidden shadow-sm sm:rounded-lg">
                        <div className="p-6 text-gray-900 dark:text-gray-100">
                            <AdminLayout user={auth.user}>
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
                                                    <div className="neon-glow-name">
                                                        {user.name}
                                                    </div>
                                                </td>
                                                <td
                                                    className={`px-6 py-4 whitespace-nowrap text-blue-500`}
                                                >
                                                    <a
                                                        href={`users/info/${user.id}`}
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
                                                    {user.roles}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    {user.orders.length}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <div className="neon-glow-edit">
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
                            </AdminLayout>
                        </div>
                    </div>
                </div>
            </div>
        </AuthenticatedLayout>
    );
}
