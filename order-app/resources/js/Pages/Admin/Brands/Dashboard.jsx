import "@/Pages/Dashboard.css";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";

import { Head } from "@inertiajs/react";

import { useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { set, useForm } from "react-hook-form";

import axios from "axios";

const BrandModal = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [submitting, setSubmitting] = useState(false);

    const openModal = () => {
        setIsOpen(true);
    };

    const closeModal = () => {
        setIsOpen(false);
    };

    const {
        register,
        handleSubmit,
        watch,
        reset,
        formState: { errors },
    } = useForm();

    const onSubmit = async (data) => {
        console.log(data);

        setSubmitting(true);

        const formData = new FormData();

        formData.append("name", data.name);
        formData.append("url", data.url);
        formData.append("description", data.description);
        formData.append("image", data.image[0]);

        try {
            const res = await axios.post(
                route("admin.create_brand"),
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }
            );

            const data = res.data;

            if (data.type === "success") {
                document.getElementById("success_modal").showModal();
                reset();
                closeModal();
            }

            if (data.type === "error") {
                console.log(data.message);
            }
        } catch (error) {
            console.log(error);
        }

        setSubmitting(false);
    };

    return (
        <>
            <dialog id="success_modal" className="modal">
                <div className="modal-box">
                    <div role="alert" className="alert alert-success">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="stroke-current shrink-0 h-6 w-6"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                        </svg>
                        <span>Your brand has been saved!</span>
                    </div>
                </div>
                <form method="dialog" className="modal-backdrop">
                    <button>close</button>
                </form>
            </dialog>

            <dialog id="error_modal" className="modal">
                <div className="modal-box">
                    <div role="alert" className="alert alert-error">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="stroke-current shrink-0 h-6 w-6"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                        </svg>
                        <span>Your brand has not been saved!</span>
                    </div>
                </div>
                <form method="dialog" className="modal-backdrop">
                    <button>close</button>
                </form>
            </dialog>

            {/* Button to open the modal */}
            <button
                className="ml-5 btn rounded-full text-white bg-yellow-700 hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-300"
                onClick={openModal}
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
            </button>

            {/* Modal Background */}
            {isOpen && (
                <div className="fixed top-0 left-0 w-full h-full bg-gray-900 bg-opacity-50 flex justify-center items-center">
                    {/* Modal Content */}
                    <div className="bg-black p-8 rounded shadow-md w-2/3">
                        <h2 className="text-xl font-bold mb-4">Insert Data</h2>
                        {/* Form to insert data */}
                        <form onSubmit={handleSubmit(onSubmit)}>
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
                                    {...register("name", { required: true })}
                                    autoComplete="off"
                                    className="input input-bordered w-full max-w-3xl"
                                    placeholder="Enter Name"
                                />
                                {errors.name && <p>This field is required</p>}
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
                                    {...register("url", { required: false })}
                                    className="input input-bordered w-full max-w-3xl"
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
                                    {...register("description", {
                                        required: false,
                                    })}
                                    className="textarea textarea-bordered textarea-lg w-4/5"
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
                                    {...register("image", { required: false })}
                                    accept="*.jpg, *.jpeg, *.png"
                                    className="file-input file-input-bordered file-input-warning w-full max-w-3xl"
                                />
                            </div>
                            <div className="text-right py-3">
                                {/* Button to submit form */}

                                {submitting && (
                                    <span className="loading loading-bars loading-lg"></span>
                                )}

                                {!submitting && (
                                    <>
                                        <button
                                            type="submit"
                                            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
                                        >
                                            Submit
                                        </button>
                                        {/* Button to close the modal */}
                                        <button
                                            className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded ml-2"
                                            type="button"
                                            onClick={closeModal}
                                        >
                                            Close
                                        </button>
                                    </>
                                )}
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </>
    );
};

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
                // <h2 className="prose font-semibold text-xl text-gray-800 dark:text-gray-200 leading-tight">
                <h2 className="prose">
                    <span className="neon--heading">Brands Dashboard</span>
                    <BrandModal />
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
