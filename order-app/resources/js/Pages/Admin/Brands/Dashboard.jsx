import "@/Pages/Dashboard.css";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";

import { Head } from "@inertiajs/react";

import { useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { Link } from "@inertiajs/react";

export default function Dashboard({ auth }) {
    const [brands, setBrands] = useState([]);

    const getBrands = useQuery({
        queryKey: ["getBrands"],
        queryFn: async () => {
            try {
                const res = await fetch(route("get_brands"));
                const data = await res.json();

                setBrands(data.brands);

                return data;
            } catch (error) {
                console.log(error);
            }
        },
    });

    return (
        <AuthenticatedLayout
            user={auth.user}
            header={
                <>
                    <h2 className="prose">
                        <span className="neon--heading">Brands Dashboard</span>
                    </h2>

                    <Link
                        href={route("admin.brands.new")}
                        className="ml-5 btn rounded-full text-white bg-yellow-700 hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-300"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-6 w-6"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                            />
                        </svg>
                        Add Brand
                    </Link>
                </>
            }
        >
            <Head title="Products Dashboard" />

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
                                            Description
                                        </th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                            Thumbnail
                                        </th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                            Actions
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="bg-dark divide-y divide-gray-200">
                                    {brands.map((brand) => (
                                        <tr key={brand.id}>
                                            <td className="px-6 py-4 whitespace-nowrap text-purple-600">
                                                <div className="--neon--purple">
                                                    {brand.name}
                                                </div>
                                            </td>
                                            <td
                                                className={`px-6 py-4 whitespace-nowrap text-blue-500`}
                                            >
                                                <span className="--neon--blue">
                                                    {brand.description}
                                                </span>
                                            </td>
                                            <td
                                                className={`px-6 py-4 whitespace-nowrap`}
                                            >
                                                <img
                                                    src={brand.image}
                                                    alt={brand.name}
                                                    className="w-16 h-16"
                                                />
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-red-600">
                                                <div>
                                                    <a
                                                        href="#"
                                                        className="px-2"
                                                    >
                                                        edit
                                                    </a>
                                                    <a
                                                        href="#"
                                                        className="px-2"
                                                    >
                                                        delete
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
