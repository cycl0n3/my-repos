<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <title>CarWash App</title>
</head>
<body>
    <header>
        <!-- Your header content here -->
    </header>

    <main>
        {{ $slot }}
    </main>

    <footer>
        <!-- Your footer content here -->
    </footer>
</body>
</html>
