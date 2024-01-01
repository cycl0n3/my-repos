import React, { useState } from "react";

import axios from "axios";

import { useForm } from "react-hook-form";

import { Head } from "@inertiajs/react";

import AuthenticatedLayout from "@/Layouts/AuthenticatedLayout";

const New = ({ auth }) => {
    const [submitting, setSubmitting] = useState(false);

    const [toastProps, setToastProps] = useState({
        type: "",
        message: "",
        visible: false,
    });

    const {
        register,
        handleSubmit,
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
                reset();
            }

            if (data.type === "error") {
                console.log(data.message);
            }

            toast(data.type, data.message);
        } catch (error) {
            console.log(error);

            toast("error", "Something went wrong. Please try again.");
        }

        setSubmitting(false);
    }

    const toast = (type, message) => {
        setToastProps({
            type,
            message,
            visible: true,
        });

        setTimeout(() => {
            setToastProps({
                type: "",
                message: "",
                visible: false,
            });
        }, 3000);
    }

    return (
        <AuthenticatedLayout
            user={auth.user}
            header={
                <>
                    <h2 className="prose">
                        <span className="neon--heading">Add Brand</span>
                    </h2>
                </>
            }
        >
            <Head title="Add Brand" />

            {/* Toast */}
            {toastProps.visible && (
                <div className="toast toast-center toast-middle">
                    <div className={`alert alert-${toastProps.type}`}>
                        <span>{toastProps.message}</span>
                    </div>
                </div>
            )}

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
                        disabled={submitting}
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
                        disabled={submitting}
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
                        disabled={submitting}
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
                        disabled={submitting}
                        {...register("image", { required: false })}
                        accept="*.jpg, *.jpeg, *.png"
                        className="file-input file-input-bordered file-input-warning w-full max-w-3xl"
                    />
                </div>
                <div className="text-center py-3">
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
                        </>
                    )}
                </div>
            </form>
        </AuthenticatedLayout>
    );
};

export default New;
