import "@/Pages/Dashboard.css";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";

import { Head } from "@inertiajs/react";

import manager from "../../../images/the-manager.svg";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

export default function Dashboard({ auth }) {
    const [totalUsers, setTotalUsers] = useState("???");
    const [totalProducts, setTotalProducts] = useState("???");
    const [totalBrands, setTotalBrands] = useState("???");

    const userCount = useQuery({
        queryKey: ["userCount"],
        queryFn: async () => {
            const res = await fetch("/user_count");
            const data = await res.json();

            setTotalUsers(data.user_count);

            return data;
        },
    });

    const productCount = useQuery({
        queryKey: ["productCount"],
        queryFn: async () => {
            const res = await fetch("/product_count");
            const data = await res.json();

            setTotalProducts(data.product_count);

            return data;
        },
    });

    const brandCount = useQuery({
        queryKey: ["brandCount"],
        queryFn: async () => {
            const res = await fetch("/brand_count");
            const data = await res.json();

            setTotalBrands(data.brand_count);

            return data;
        },
    });

    return (
        <AuthenticatedLayout
            user={auth.user}
            header={
                <h2 className="prose">
                    <p className="neon--heading">Admin Dashboard</p>
                </h2>
            }
        >
            <Head title="Admin Dashboard" />

            <section class="py-10 bg-white sm:py-16 lg:py-24">
                <div class="px-4 mx-auto max-w-7xl sm:px-6 lg:px-8">
                    <div class="text-center">
                        <h4 class="text-xl font-medium text-gray-900">
                            Numbers tell the hard works we’ve done in last 6
                            years
                        </h4>
                    </div>

                    <div class="grid grid-cols-1 gap-6 px-6 mt-8 sm:px-0 lg:mt-16 sm:grid-cols-2 lg:grid-cols-4 xl:gap-x-12">
                        <div class="overflow-hidden bg-white border border-gray-200 rounded-lg">
                            <div class="px-4 py-6">
                                <div class="flex items-start">
                                    <svg
                                        class="flex-shrink-0 w-12 h-12 text-fuchsia-600"
                                        xmlns="http://www.w3.org/2000/svg"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            stroke-width="1"
                                            d="M13 10V3L4 14h7v7l9-11h-7z"
                                        />
                                    </svg>
                                    <div class="ml-4">
                                        <h4 class="text-4xl font-bold text-gray-900">
                                            {totalBrands}
                                        </h4>
                                        <p class="mt-1.5 text-lg font-medium leading-tight text-gray-500">
                                            Total Brands
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="overflow-hidden bg-white border border-gray-200 rounded-lg">
                            <div class="px-4 py-6">
                                <div class="flex items-start">
                                    <svg
                                        class="flex-shrink-0 w-12 h-12 text-fuchsia-600"
                                        xmlns="http://www.w3.org/2000/svg"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            stroke-width="1"
                                            d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"
                                        />
                                    </svg>
                                    <div class="ml-4">
                                        <h4 class="text-4xl font-bold text-gray-900">
                                            {totalUsers}
                                        </h4>
                                        <p class="mt-1.5 text-lg font-medium leading-tight text-gray-500">
                                            Happy customers
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="overflow-hidden bg-white border border-gray-200 rounded-lg">
                            <div class="px-4 py-6">
                                <div class="flex items-start">
                                    <svg
                                        class="flex-shrink-0 w-12 h-12 text-fuchsia-600"
                                        xmlns="http://www.w3.org/2000/svg"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path d="M11.5 23l-8.5-4.535v-3.953l5.4 3.122 3.1-3.406v8.772zm1-.001v-8.806l3.162 3.343 5.338-2.958v3.887l-8.5 4.534zm-10.339-10.125l-2.161-1.244 3-3.302-3-2.823 8.718-4.505 3.215 2.385 3.325-2.385 8.742 4.561-2.995 2.771 2.995 3.443-2.242 1.241v-.001l-5.903 3.27-3.348-3.541 7.416-3.962-7.922-4.372-7.923 4.372 7.422 3.937v.024l-3.297 3.622-5.203-3.008-.16-.092-.679-.393v.002z" />
                                    </svg>
                                    <div class="ml-4">
                                        <h4 class="text-4xl font-bold text-gray-900">
                                            {totalProducts}
                                        </h4>
                                        <p class="mt-1.5 text-lg font-medium leading-tight text-gray-500">
                                            Total Products
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="overflow-hidden bg-white border border-gray-200 rounded-lg">
                            <div class="px-4 py-6">
                                <div class="flex items-start">
                                    <svg
                                        class="flex-shrink-0 w-12 h-12 text-fuchsia-600"
                                        xmlns="http://www.w3.org/2000/svg"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            stroke-width="1"
                                            d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"
                                        />
                                    </svg>
                                    <div class="ml-4">
                                        <h4 class="text-4xl font-bold text-gray-900">
                                            98%
                                        </h4>
                                        <p class="mt-1.5 text-lg font-medium leading-tight text-gray-500">
                                            Customer success
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </AuthenticatedLayout>
    );
}
