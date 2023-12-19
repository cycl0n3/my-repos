import "@/Pages/Dashboard.css";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";

import { Head } from "@inertiajs/react";

import manager from "../../../images/the-manager.svg";
import worker from "../../../images/the-worker.svg";

export default function UsersDashboard({ auth, users }) {
    return (
        <AuthenticatedLayout
            user={auth.user}
            header={
                <h2 className="prose">
                    <p className="neon--heading">Users Dashboard</p>
                </h2>
            }
        >
            <Head title="Users Dashboard" />

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
                                            Actions
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="bg-dark divide-y divide-gray-200">
                                    {users.map((user) => (
                                        <tr key={user.id}>
                                            <td className="px-6 py-4 whitespace-nowrap text-purple-600">
                                                <div className="--neon--purple">
                                                    {user.name}
                                                </div>
                                            </td>
                                            <td
                                                className={`px-6 py-4 whitespace-nowrap text-blue-500`}
                                            >
                                                <a
                                                    href={`users/info/${user.id}`}
                                                    className="--neon--blue"
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
                                                        user.roles === "ADMIN"
                                                            ? manager
                                                            : worker
                                                    }
                                                    alt="roles"
                                                    className="w-12 h-12 inline-block"
                                                />
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-red-600">
                                                <div className="--neon--red">
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
        </AuthenticatedLayout>
    );
}
