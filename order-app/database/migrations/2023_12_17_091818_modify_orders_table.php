<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('orders', function (Blueprint $table) {
            // Drop the foreign key constraint
            $table->dropForeign('orders_product_id_foreign');

            // Remove product_id column
            $table->dropColumn('product_id');

            // Change order_number column to UUID
            $table->uuid('order_number')->change();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('orders', function (Blueprint $table) {
            // To rollback changes if needed
            $table->unsignedBigInteger('product_id');
            $table->bigIncrements('order_number')->change();
        });
    }
};
