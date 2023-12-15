import './bootstrap';

import '../css/app.css';

import { createRoot } from 'react-dom/client';
import { createInertiaApp } from '@inertiajs/react';
import { resolvePageComponent } from 'laravel-vite-plugin/inertia-helpers';

const appName = import.meta.env.VITE_APP_NAME || 'Laravel';

import {
    QueryClient,
    QueryClientProvider,
} from '@tanstack/react-query';

import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

createInertiaApp({
    title: (title) => `${title} - ${appName}`,
    resolve: (name) => resolvePageComponent(`./Pages/${name}.jsx`, import.meta.glob('./Pages/**/*.jsx')),
    setup({ el, App, props }) {
        const root = createRoot(el);

        const queryClient = new QueryClient();

        const TheApp = () => (
            <QueryClientProvider client={queryClient}>
                <App {...props} />
                <ReactQueryDevtools initialIsOpen={false} />
            </QueryClientProvider>
        );

        root.render(<TheApp />);
    },
    progress: {
        color: '#4B5563',
    },
});
