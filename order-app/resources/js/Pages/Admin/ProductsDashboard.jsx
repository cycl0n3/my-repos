import "@/Pages/Dashboard.css";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";

import { Head } from "@inertiajs/react";

import { useState } from "react";

const ProductModal = () => {
    const [isOpen, setIsOpen] = useState(false);

    const openModal = () => {
        setIsOpen(true);
    };

    const closeModal = () => {
        setIsOpen(false);
    };

    return (
        <>
            {/* Button to open the modal */}
            <button
                className="ml-5 inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-yellow-700 hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-300"
                onClick={openModal}
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 mr-2"
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
                Add Product
            </button>

            {/* Modal Background */}
            {isOpen && (
                <div className="fixed top-0 left-0 w-full h-full bg-gray-900 bg-opacity-50 flex justify-center items-center">
                    {/* Modal Content */}
                    <div className="bg-black p-8 rounded shadow-md w-2/3">
                        <h2 className="text-xl font-bold mb-4">Insert Data</h2>
                        {/* Form to insert data */}
                        <form>
                            <div className="mb-4">
                                <label
                                    htmlFor="name"
                                    className="block text-gray-700 font-bold mb-2"
                                >
                                    Name
                                </label>
                                <input
                                    type="text"
                                    id="name"
                                    name="name"
                                    className="border rounded w-full p-2"
                                    placeholder="Enter Name"
                                />
                            </div>
                            <div className="mb-4">
                                <label
                                    htmlFor="url"
                                    className="block text-gray-700 font-bold mb-2"
                                >
                                    URL
                                </label>
                                <input
                                    type="text"
                                    id="url"
                                    name="url"
                                    className="border rounded w-full p-2"
                                    placeholder="Enter URL"
                                />
                            </div>
                            <div className="mb-4">
                                <label
                                    htmlFor="description"
                                    className="block text-gray-700 font-bold mb-2"
                                >
                                    Description
                                </label>
                                <textarea
                                    id="description"
                                    name="description"
                                    className="border rounded w-full p-2"
                                    placeholder="Enter Description"
                                ></textarea>
                            </div>
                            <div className="mb-4">
                                <label
                                    htmlFor="image"
                                    className="block text-gray-700 font-bold mb-2"
                                >
                                    Image
                                </label>
                                <input
                                    type="file"
                                    id="image"
                                    name="image"
                                    className="border rounded w-full p-2"
                                />
                            </div>
                            <div className="text-right">
                                {/* Button to submit form */}
                                <button
                                    type="submit"
                                    className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
                                    onClick={(e) => {
                                        e.preventDefault();
                                        closeModal();
                                    }}
                                >
                                    Submit
                                </button>
                                {/* Button to close the modal */}
                                <button
                                    className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded ml-2"
                                    onClick={closeModal}
                                >
                                    Close
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </>
    );
};

export default function UsersDashboard({ auth, products }) {
    return (
        <AuthenticatedLayout
            user={auth.user}
            header={
                <h2 className="prose">
                    <span className="neon--heading">Products Dashboard</span>
                    <ProductModal />
                </h2>
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
                                    {products.map((product) => (
                                        <tr key={product.id}>
                                            <td className="px-6 py-4 whitespace-nowrap text-purple-600">
                                                <div className="--neon--purple">
                                                    {product.name}
                                                </div>
                                            </td>
                                            <td
                                                className={`px-6 py-4 whitespace-nowrap text-blue-500`}
                                            >
                                                <span className="--neon--blue">
                                                    {product.description.substring(
                                                        0,
                                                        50
                                                    )}{" "}
                                                    ...
                                                </span>
                                            </td>
                                            <td
                                                className={`px-6 py-4 whitespace-nowrap`}
                                            >
                                                {product.url ? (
                                                    <img
                                                        src={product.url}
                                                        alt={product.name}
                                                        className="w-10 h-10 rounded-full"
                                                    />
                                                ) : (
                                                    <img
                                                        src="https://placehold.co/200"
                                                        alt={product.name}
                                                        className="w-10 h-10 rounded-full"
                                                    />
                                                )}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-red-600">
                                                <div
                                                    className="--neon--red"
                                                    style={{
                                                        display: "flex",
                                                        justifyContent:
                                                            "space-between",
                                                    }}
                                                >
                                                    <a
                                                        href={`products/edit/${product.id}`}
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
