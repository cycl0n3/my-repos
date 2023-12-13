function AdminLayout({ user, children }) {
  if (user.roles === 'ADMIN') {
    return (
      <div>
        {children}
      </div>
    );
  } else {
    return null;
  }
}

export default AdminLayout;
